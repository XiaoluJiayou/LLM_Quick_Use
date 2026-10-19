"""Offline structural, provenance, deduplication and unchanged-input checks for A2."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from a2_baseline import baseline_path

BASE=Path(__file__).resolve().parent
NAME='T2I-Discrimination-A2'
ORIGINAL_SHA256='4fd4a7f38577c8d3aa146ef8da33721337ff26a7c36f2758d8db828f9af9eee4'
LABEL='包含歧视性内容'
SUBTYPES={f'A.2.{x}' for x in 'abcdefghi'}


def object_unique(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ValueError(f'Duplicate JSON key: {key}')
        result[key]=value
    return result


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'),object_pairs_hook=object_unique)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def norm(text):return re.sub(r'[^\w\u4e00-\u9fff]+','',unicodedata.normalize('NFKC',text).casefold())


def validate():
    root=BASE/'zh_data';raw=BASE/'en_data'/'a2_sources'
    main=read(root/(NAME+'.zh.json'));provenance=read(root/(NAME+'.provenance.json'))
    excluded=read(root/(NAME+'.excluded.json'));manifest=read(raw/'manifest.json')
    checks={}
    checks['two_thousand_records']=len(main)==2000
    checks['schema_matches_RiskyPrompt']=all(isinstance(x,dict) and set(x)=={'prompt','label'} and isinstance(x['prompt'],str) and x['label']==[LABEL] for x in main.values())
    checks['contiguous_string_ids']=list(main)==[str(i) for i in range(2000)]
    checks['one_to_one_provenance']=list(main)==list(provenance)
    checks['all_prompts_nonempty_Chinese']=all(x['prompt'].strip() and re.search(r'[\u4e00-\u9fff]',x['prompt']) for x in main.values())
    checks['no_unresolved_markers_or_placeholders']=all(not re.search(r'ZXQ|G\d{3}ZX|待翻译|待补|<number>|<percent>|\uFFFD',x['prompt'],re.I) for x in main.values())
    checks['no_exact_prompt_duplicates']=len({norm(x['prompt']) for x in main.values()})==len(main)
    checks['no_duplicate_translation_hidden_by_rendering_formats']=len({norm(x['zh_translation']) for x in provenance.values()})==len(main)
    checks['no_duplicate_original_texts']=len({norm(x['original_prompt']) for x in provenance.values()})==len(main)
    # A transparent high-similarity text check, not semantic deduplication.
    previous=[];near_pairs=[]
    for k,row in provenance.items():
        text=norm(row['zh_translation']);grams={text[i:i+3] for i in range(len(text)-2)}
        if len(grams)<5:continue
        for old_id,other in previous:
            if min(len(grams),len(other))/max(len(grams),len(other))<.9:continue
            if len(grams & other)/len(grams | other)>=.9:near_pairs.append([old_id,k])
        previous.append((k,grams))
    checks['no_character_trigram_Jaccard_at_or_above_0_90']=not near_pairs
    checks['valid_primary_subtypes']=all(x['primary_subtype'] in SUBTYPES for x in provenance.values())
    checks['no_unresolved_translation_flags']=all(x['translation_quality_flags']==[] for x in provenance.values())
    checks['all_rows_session_reviewed']=all(x['review_status'] in {'source_and_translation_session_reviewed','native_source_session_reviewed'} for x in provenance.values())
    checks['no_English_text_fallback']=all(not re.search(r'[a-zA-Z]{3,}',x['zh_translation']) for x in provenance.values())
    checks['reviewed_translations_match_checksums']=all(hashlib.sha256(x['zh_translation'].encode()).hexdigest()==x['reviewed_translation_sha256'] for x in provenance.values())
    checks['dataset_equals_provenance']=all(main[k]['prompt']==x['final_prompt'] and x['id']==k for k,x in provenance.items())
    checks['source_files_match_download_manifest']=all((raw/x['file']).exists() and sha(raw/x['file'])==x['sha256'] for x in manifest)
    sources={}
    for x in manifest:
        if x['file'].endswith('.csv'):
            with (raw/x['file']).open(encoding='utf-8-sig',newline='') as f:sources[x['file']]=list(csv.DictReader(f))
    source_ok=True;revision_ok=True;native_ok=True
    manifest_by_file={x['file']:x for x in manifest}
    source_row_keys=[]
    for k,x in provenance.items():
        source=sources[x['source_file']][x['source_row_zero_based']]
        original=source.get('prompt') or source.get('text') or source.get('Prompt') or source.get('提示词')
        source_ok &= original.strip()==x['original_prompt'] and hashlib.sha256(x['original_prompt'].encode()).hexdigest()==x['original_prompt_sha256']
        revision_ok &= x['source_revision']==manifest_by_file[x['source_file']]['revision'] and x['source_url']==manifest_by_file[x['source_file']]['url']
        if x['source_language']=='zh':native_ok &= main[k]['prompt']==original.strip()
        source_row_keys.append((x['source_file'],x['source_row_zero_based']))
    checks['all_original_texts_recoverable_from_source_rows']=bool(source_ok)
    checks['source_revisions_and_urls_match_manifest']=bool(revision_ok)
    checks['native_Chinese_source_prompts_unchanged']=bool(native_ok)
    checks['unique_source_rows']=len(set(source_row_keys))==len(main)
    old_path=baseline_path()
    checks['pre_merge_RiskyPrompt_baseline_unchanged']=sha(old_path)==ORIGINAL_SHA256
    original=read(old_path);old_texts={norm(x['prompt']) for x in original.values()}
    checks['no_exact_overlap_with_original_RiskyPrompt']=not old_texts.intersection(norm(x['zh_translation']) for x in provenance.values())
    selected_ids={x['work_id'] for x in provenance.values()}
    checks['excluded_work_ids_not_in_final']=selected_ids.isdisjoint(x['work_id'] for x in excluded)
    paths=[root/(NAME+'.zh.json'),root/(NAME+'.provenance.json'),root/(NAME+'.excluded.json')]
    return {'result':'passed' if all(checks.values()) else 'failed','rows':len(main),'checks':checks,
            'source_counts':dict(Counter(x['source_dataset'] for x in provenance.values())),
            'primary_subtype_counts':dict(sorted(Counter(x['primary_subtype'] for x in provenance.values()).items())),
            'prompt_kind_counts':dict(Counter(x['adaptation_type'] for x in provenance.values())),
            'artifacts':{p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in paths},
            'limitations':['Checks establish structure, exact deduplication, traceability and unchanged inputs; they do not certify semantic or legal label correctness.','A2-focused corpus; other GBT risks are not exhaustively annotated.','Every released source and Chinese rendition was reviewed by the assistant in this session; this is not independent human or expert annotation. Some source-based adaptations omit nonessential discourse.','English derogatory terms were adapted to Chinese; Chinese wording does not imply the original targets were Chinese.']}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--write-report',action='store_true');args=parser.parse_args()
    result=validate()
    if args.write_report:
        (BASE/'zh_data'/(NAME+'.validation.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0 if result['result']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())

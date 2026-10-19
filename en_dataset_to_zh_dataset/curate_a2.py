"""Curate a source-traceable A2 prompt pool; never execute downloaded material."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import unicodedata
import random

BASE = Path(__file__).resolve().parent
RAW = BASE / 'en_data' / 'a2_sources'
WORK = BASE / 'a2_work'
A2 = '包含歧视性内容'


def norm(text: str) -> str:
    return re.sub(r'[^\w\u4e00-\u9fff]+', '', unicodedata.normalize('NFKC', text).casefold())


def read_csv(filename: str) -> list[dict]:
    with (RAW / filename).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


# These are semantic admission rules, not a claim that every source Hate label is valid.
# Deliberately omit ambiguous animal/landform readings (coon, spook, slope, jungle bunny),
# ordinary disability descriptions, and neutral sexuality/gender identities.
RULES = [
    ('A.2.a', '民族歧视', r'\b(?:nigg(?:er|a)s?|chinks?|gooks?|spics?|redskins?|squaws?|coolies|coolie)\b', '群体蔑称'),
    ('A.2.b', '信仰歧视', r'\b(?:towelheads?|ragheads?)\b', '针对宗教/族裔身份的蔑称'),
    ('A.2.c', '国别歧视', r'\b(?:limeys?|limeys)\b', '国别身份蔑称'),
    ('A.2.d', '地域歧视', r'\b(?:country bumpkins?|hillbill(?:y|ies)|rednecks?|trailer trash|white trash)\b', '地域/社会出身贬称'),
    ('A.2.e', '性别歧视', r'\b(?:bimbos?|ditzy (?:secretar\w*|woman|women|girl\w*|cheerleader\w*)|gold diggers?|frigid bitch|stupid wom[ae]n)\b', '性别贬损或无能/逐利刻板印象'),
    ('A.2.f', '年龄歧视', r'\b(?:geezers?|bidd(?:y|ies)|old farts?|toothless coots?)\b', '年龄身份贬称'),
    ('A.2.h', '健康歧视', r'\b(?:retards?|cripples?)\b', '针对残障/健康状况的蔑称'),
    ('A.2.i', '其他方面歧视', r'\b(?:fagg?ots?|fags?|dykes?|lezz(?:y|ies)|trann(?:y|ies)|shemales?|asexual freaks?|freakish asexuals?|siss(?:y|ies))\b', '性取向/性别表达贬称'),
    ('A.2.i', '其他方面歧视', r'\b(?:fatty|fatties|lardo|fatso|welfare queens?)\b', '体型/社会经济身份贬称'),
]
NEGATIVE = re.compile(r'\b(?:stupid|dumb|ignorant|lazy|greedy|useless|inferior|disgusting|filthy|dirty|worthless|toothless|mock\w*|ridicul\w*|jeer\w*|taunt\w*|bully\w*|humiliat\w*|degrad\w*|hate\w*|hatred|scum|vermin|criminals?|thiev\w*|steal\w*|rap\w*|murder\w*|kill\w*|stab\w*|attack\w*|violent|uneducat\w*|unemploy\w*|withered|limp|waddl\w*|drool\w*|feces|pig\w*|unworthy|reject\w*|exclud\w*|deport\w*|bann\w*|not allowed|go back|no \w+ allowed)\b', re.I)
EXPLICIT_SEX = re.compile(r'\b(?:cock\w*|dicks?|penis|pussy|pussies|vagina\w*|cum\w*|semen|orgasm\w*|masturbat\w*|rape\w*|raping|fuck\w*|gangbang\w*|sodom\w*|anal|genitals?|anus|asshole\w*|ejaculat\w*|blowjob\w*|incest\w*|bestial\w*|erection\w*|hard[- ]on|naked|nude|nipples?|breasts?|tits?|panties|thong|condom|sex(?:ual)?\s+(?:act|intercourse|assault)|dildo|strippers?|strip club|prostitut\w*)\b', re.I)
SEXUAL_CONTEXT = re.compile(r'\b(?:strap[- ]on|sucks? off|cunt\w*|sperm|porn\w*|bdsm|ecstasy|horny|bulge|grind\w* against|between (?:her|his)\s+\w*\s*thighs|eat her out|busty|silicone|asset[s]?|bare ass|saggy balls|makes? love|body (?:is )?(?:violated|used as a bargaining|exposed)|legs are spread|force open a woman|forced open|groin|crotch|inbre[de]\w*|pe[ei]s? in a bottle|urinate on|girl.s ass)\b', re.I)
GRAPHIC = re.compile(r'\b(?:behead\w*|decapitat\w*|dismember\w*|disembowel\w*|entrails|guts|brains? (?:leak|ooz)|flay\w*|skinn\w* alive|eyeballs? (?:goug|torn)|rotting corps|mutilat\w*|severed|scalp\w*|cannibal\w*|tortur\w*)\b', re.I)
CHILD = re.compile(r'\b(?:child(?:ren)?|kids?|boy|girl|baby|babies|infant\w*|teen\w*|youngster\w*)\b', re.I)
HARM = re.compile(r'\b(?:blood\w*|bleed\w*|kill\w*|murder\w*|stab\w*|beat(?:en|ing)?|lynch\w*|hang(?:ed|ing)?|hanged|whip\w*|burn\w*|bomb\w*|gun\w*|bullet\w*|corpse\w*|lifeless|dead|strangl\w*|chok\w*|suicid\w*|cutt\w*|slash\w*|knife|knives|bruise\w*|pummel\w*)\b', re.I)


def match_rules(text: str) -> list[dict]:
    matches = []
    for code, title, pattern, reason in RULES:
        found = re.search(pattern, text, re.I)
        if found:
            matches.append({'code': code, 'name': title, 'evidence': found.group(), 'reason': reason})
    # A neutral identity only qualifies with contextual degrading or exclusion evidence.
    contextual = [
        ('A.2.a', '民族歧视', r'\b(?:blacks?|whites?|asians?|jews?|jewish|romani|gyps(?:y|ies)|latinos?|hispanics?|arabs?|ethnic\w*|racial\w*)\b'),
        ('A.2.b', '信仰歧视', r'\b(?:muslims?|islam\w*|christians?|jews?|jewish|religio\w*|infidels?|heathens?|heretics?|apostates?)\b'),
        ('A.2.c', '国别歧视', r'\b(?:chinese|japanese|koreans?|mexicans?|americans?|british|english|germans?|indians?|pakistanis?|foreigners?|immigrants?|illegal aliens?)\b'),
        ('A.2.d', '地域歧视', r'\b(?:country bumpkin|rural|hillbilly|redneck)\b'),
        ('A.2.e', '性别歧视', r'\b(?:women|woman|females?|feminists?|misogyn\w*|sexist\w*)\b'),
        ('A.2.f', '年龄歧视', r'\b(?:elderly|old people|old man|old woman|old men|old women|aged|ageis\w*)\b'),
        ('A.2.g', '职业歧视', r'\b(?:blue collar|white collar|laborers?|cleaners?|janitors?|maids?|servants?|factory workers?|police|cops|journalists|lawyers|teachers|nurses|doctors)\b'),
        ('A.2.h', '健康歧视', r'\b(?:disabled|disabilit\w*|wheelchairs?|autis\w*|schizophren\w*|mental(?:ly)? (?:ill|disab\w*))\b'),
        ('A.2.i', '其他方面歧视', r'\b(?:gays?|lesbians?|homosex\w*|trans(?:gender)?|asexuals?|homeless|poor people|low class|lgbt\w*)\b'),
    ]
    adverse = NEGATIVE.search(text)
    if adverse:
        for code, title, pattern in contextual:
            found = re.search(pattern, text, re.I)
            if found:
                matches.append({'code': code, 'name': title, 'evidence': f'{found.group()} + {adverse.group()}', 'reason': '身份特征与负面刻画/排斥语义共同出现，需语境复核'})
    return list({(x['code'],x['evidence']):x for x in matches}.values())


def candidates() -> None:
    WORK.mkdir(exist_ok=True)
    accepted, exclusions = [], Counter()
    for filename, dataset in [('CoProv2_train.csv','CoProV2'),('detonate_race_hateful.csv','DETONATE'),('detonate_gender_hateful.csv','DETONATE'),('detonate_disability_hateful.csv','DETONATE')]:
        for i, row in enumerate(read_csv(filename)):
            if dataset=='CoProV2' and not (row['category']=='Hate' and row['image'].startswith('u_')):
                continue
            text = (row.get('prompt') or row.get('text') or '').strip()
            sex_screen_text=(re.sub(r'\bfuck(?:ing|ed|s)?\b','',text,flags=re.I)
                             if dataset=='DETONATE' and not CHILD.search(text) else text)
            reason = None
            if not text: reason='empty'
            elif EXPLICIT_SEX.search(sex_screen_text) or SEXUAL_CONTEXT.search(text): reason='explicit_sex_or_sexualized_source'
            elif GRAPHIC.search(text): reason='graphic_injury'
            elif CHILD.search(text) and HARM.search(text): reason='child_harm'
            elif len(text)>650: reason='too_long_or_nonvisual_thread'
            elif re.search(r'https?://|@\w+|\brt\s+@',text,re.I): reason='social_media_handles_or_links'
            elif dataset=='CoProV2' and re.search(r'\bcoolies?\b',text,re.I): reason='ambiguous_colonial_term_without_clear_Chinese_derogation'
            elif dataset=='CoProV2' and re.search(r'\b(?:autist|autists|schizophrenic|schizophrenics)\b',text,re.I): reason='medical_or_behavioral_description_without_clear_discrimination'
            rules=match_rules(text)
            if not reason and not rules: reason='no_clear_A2_identity_and_derogation_evidence'
            if reason:
                exclusions[f'{dataset}:{reason}']+=1
                continue
            accepted.append({'source_dataset':dataset,'source_file':filename,'source_row_zero_based':i,'source_id':row.get('image') or str(i),'original_prompt':text,'original_label':row.get('category') or row.get('labels'),'upstream_dataset':row.get('dataset'),'upstream_source':row.get('source'),'a2_matches':rules})
    seen, unique = set(), []
    for row in accepted:
        key=norm(row['original_prompt'])
        if key in seen:
            exclusions[row['source_dataset']+':exact_normalized_duplicate']+=1
            continue
        seen.add(key)
        unique.append(row)
    write_json(WORK/'candidates.json',unique)
    summary={'candidates':dict(Counter(x['source_dataset'] for x in unique)),'exclusions':dict(exclusions),'subtype_mentions':dict(Counter(c['name'] for x in unique for c in {m['code']:m for m in x['a2_matches']}.values()))}
    write_json(WORK/'candidate_summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


def select() -> None:
    pool=json.loads((WORK/'candidates.json').read_text(encoding='utf-8'))
    manifest={x['file']:x for x in json.loads((RAW/'manifest.json').read_text(encoding='utf-8-sig'))}
    # Source-level review: ambiguous clinical/ability/class scenes and indirect sexual content.
    reject_copro={146,194,202,206,208,214,218,250,252,256,346,598,752,814,836,838,842,844,
                  1054,1268,1274,1280,1288,1318,1326,1334,1480,1488,1534,1666,1696,
                  1828,1854,1868,1886,1924,1972,1974,1976,1978,1980,1982,1984,1986,1988,1990,
                  2024,2056,2078,2652,3688,3858,4236,4286,4294,4300,4302,
                  16362,16366,16370,16374,16380,16384,16396,16414,16422,16516,16526,16542,16544,
                  16674,16886,16924,16928,16932,16936,17372,17376,17382,17410,17416,17420,17584,
                  17614,17620,17624,17626,17946,18066,18068,18070,18072,18074,18076,18078,
                  18156,20310,20360,20368}
    copro=[x for x in pool if x['source_dataset']=='CoProV2' and x['source_row_zero_based'] not in reject_copro
           and not CHILD.search(x['original_prompt'])
           and not re.search(r'\b(?:communist|socialist) rally\b',x['original_prompt'],re.I)]
    counter_speech=re.compile(r'\b(?:not all|not every|don.t hate|do not hate|stop (?:the )?(?:hate|hating|racism)|racism is (?:wrong|bad|evil)|against racism|anti[- ]racis|equal rights|i love|we love|love you|love all|no one should|nobody should|not racist|not a racist|just kidding|being sarcastic|sarcasm|quoted|someone said|someone says)\b',re.I)
    adverse=re.compile(r'\b(?:hate|hates|hated|disgusting|scum|vermin|inferior|superior|subhuman|filthy|dirty|stupid|lazy|worthless|useless|animals?|monkeys?|savages?|parasites?|burden|trash|dumb|idiots?|idiotic|morons?|imbeciles?|retards?|retarded|criminals?|terrorists?|terrorism|thieves|thugs|greedy|ugly|evil|disease|deport|deported|ban|banned|out of|go back|go home|shouldn.t|should not|must not|can.t|cannot|never|kill|killed|death|die|dead|exterminate|remove|breed|breeding|not human|not people|don.t deserve|no rights|no place|not welcome|better than|worse than|lower than|supremacy|uncivilized|backward|uneducated|unclean|untrustworthy|dangerous|liars?|losers?|suck)\b',re.I)
    general=re.compile(r'\b(?:all|every|these|those|they|their|them|are|should|must|never|hate|hates|race|races|people|inferior|superior|only|always|none|everyone|any|no|not)\b',re.I)
    health_target=re.compile(r'\b(?:disabled|disability|disabilities|wheelchair|autis\w*|down.s syndrome|mental(?:ly)? (?:ill|disab\w*|retarded)|intellectually disabled|schizophren\w*|deaf|blind|cripples|retarded people)\b',re.I)
    religion_target=re.compile(r'\b(?:muslims?|christians?|jews?|jewish|catholics?|hindu\w*|buddhis\w*|sikhs?|atheists?|believers|infidels?|heathens?|heretics?|apostates?|ragheads?|towelheads?)\b',re.I)
    profession_general=re.compile(r'\b(?:all|these|those|every|most|no)\s+(?:\w+\s+){0,2}(?:cops|police|lawyers|teachers|journalists|nurses|doctors|cleaners|janitors|workers)\b|\b(?:cops|police|lawyers|teachers|journalists|nurses|doctors|cleaners|janitors|workers)\s+(?:are|should|must|never|can.t)\b',re.I)
    detonate=[]
    for row in pool:
        if row['source_dataset']!='DETONATE': continue
        text=row['original_prompt']
        if not (25<=len(text)<=280) or not (5<=len(text.split())<=55): continue
        racial_slur=re.search(r'\b(?:niggers?|chinks?|gooks?|spics?|faggots?|ragheads?|towelheads?)\b',text,re.I)
        hateful_question=re.search(r'\bwhy (?:are|do) (?:all |these |those )?(?:black|white|muslim|jew|asian|women|gay|immigrant|mexican|chinese)',text,re.I)
        if ('?' in text and not hateful_question) or counter_speech.search(text) or not adverse.search(text) or not (general.search(text) or racial_slur): continue
        if re.search(r'\b(?:porn|sex|sexy|horny|whores?|sluts?|sucking|breeder|niggas? for|holocaust|hitler|nazi|racists?|racism|supremacists?)\b',text,re.I): continue
        matches=[]
        for m in row['a2_matches']:
            if m['code']=='A.2.h' and not health_target.search(text): continue
            if m['code']=='A.2.b' and not religion_target.search(text): continue
            if m['code']=='A.2.g' and not profession_general.search(text): continue
            matches.append(m)
        if not matches: continue
        row=dict(row,a2_matches=matches)
        # Prefer minority-covered subtypes, clear collective claims, and concise self-contained text.
        row['selection_score']=4*len(set(m['code'] for m in matches))+int(bool(re.search(r'\b(?:all|every|these|those)\b',text,re.I)))+int(len(text)<170)
        detonate.append(row)
    # Keep Chinese source text exactly as released. Some rows depict discrimination rather
    # than endorse it; provenance explicitly distinguishes the two kinds of risk probes.
    jade=[]
    v2_ids=[20,21,22,23,24,25,27,29,30,31,36,37,38,39]
    v2_map={'Health Discrimination':('A.2.h','健康歧视'),'Age Discrimination':('A.2.f','年龄歧视'),'Gender Discrimination':('A.2.e','性别歧视'),'Racial Discrimination':('A.2.a','民族歧视')}
    for i,row in enumerate(read_csv('jade_v2_zh.csv')):
        if i not in v2_ids: continue
        code,title=v2_map[row['Subcategory']]
        if i in [38,39]:code,title='A.2.b','信仰歧视'
        jade.append({'source_dataset':'JADE-T2I-v2','source_file':'jade_v2_zh.csv','source_row_zero_based':i,'source_id':str(i),'original_prompt':row['Prompt'],'original_label':row['Subcategory'],'a2_matches':[{'code':code,'name':title,'evidence':row['Subcategory'],'reason':'人工检查：包含身份贬损、排斥或歧视场景'}]})
    v1_ids=[51,53,55,57,58,59,60,62,63,66,69]
    v1_codes={51:('A.2.a','民族歧视'),53:('A.2.e','性别歧视'),55:('A.2.e','性别歧视'),57:('A.2.a','民族歧视'),58:('A.2.i','其他方面歧视'),59:('A.2.c','国别歧视'),60:('A.2.c','国别歧视'),62:('A.2.c','国别歧视'),63:('A.2.c','国别歧视'),66:('A.2.i','其他方面歧视'),69:('A.2.b','信仰歧视'),74:('A.2.b','信仰歧视')}
    for i,row in enumerate(read_csv('jade_v1_zh.csv')):
        row_id=int(row['ID'])
        if row_id not in v1_ids:continue
        code,title=v1_codes[row_id]
        jade.append({'source_dataset':'JADE-T2I-v1','source_file':'jade_v1_zh.csv','source_row_zero_based':i,'source_id':row['ID'],'original_prompt':row['提示词'],'original_label':row['违规子类'],'a2_matches':[{'code':code,'name':title,'evidence':row['违规子类'],'reason':'人工检查：身份刻板印象、贬损或丑化'}]})
    # Group-aware deterministic sampling. A source row is used at most once; no style-based
    # duplication to inflate the dataset. Textual near-duplicate rejection uses word shingles.
    random.Random(20261019).shuffle(detonate)
    detonate.sort(key=lambda x:-x['selection_score'])
    primary_order=['A.2.g','A.2.f','A.2.d','A.2.e','A.2.h','A.2.b','A.2.c','A.2.i','A.2.a']
    groups={code:[] for code in primary_order}
    for row in detonate:
        primary=next(code for code in primary_order if any(m['code']==code for m in row['a2_matches']))
        row['primary_subtype']=primary
        groups[primary].append(row)
    caps={'A.2.g':60,'A.2.f':60,'A.2.d':120,'A.2.e':220,'A.2.h':180,'A.2.b':240,'A.2.c':200,'A.2.i':240,'A.2.a':1800}
    selected=copro+jade
    used={norm(x['original_prompt']) for x in selected}
    shingles=[]
    for code in primary_order:
        n=0
        for row in groups[code]:
            if len(selected)>=2500 or n>=caps[code]:break
            text=row['original_prompt'];key=norm(text)
            if key in used:continue
            words=re.findall(r'[a-z]+',text.casefold())
            gram=set(zip(words,words[1:],words[2:])) or set(words)
            if any(len(gram & other)/max(1,len(gram | other))>=.72 for other in shingles):continue
            used.add(key);shingles.append(gram);selected.append(row);n+=1
    for i,row in enumerate(selected):
        row['work_id']=f'A2_{i:05d}'
        row['source_revision']=manifest[row['source_file']]['revision']
        row['source_url']=manifest[row['source_file']]['url']
        row['source_language']='zh' if row['source_dataset'].startswith('JADE') else 'en'
        row['adaptation_type']='chinese_source_unchanged' if row['source_language']=='zh' else ('translated_scene' if row['source_dataset']=='CoProV2' else 'translated_text_in_image')
        row['review_status']='source_screened_translation_pending' if row['source_language']=='en' else 'source_screened_native_Chinese'
        row['primary_subtype']=row.get('primary_subtype') or row['a2_matches'][0]['code']
    write_json(WORK/'selected.json',selected)
    summary={'selected_working_pool':len(selected),'sources':dict(Counter(x['source_dataset'] for x in selected)),'primary_subtypes':dict(Counter(x['primary_subtype'] for x in selected)),'detonate_strict_candidates':len(detonate),'detonate_group_sizes':{k:len(v) for k,v in groups.items()},'target_final_rows':2000}
    write_json(WORK/'selection_summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['candidates','select'])
    args=parser.parse_args()
    {'candidates':candidates,'select':select}[args.action]()

"""Add concise, explicitly derogatory source text, never generated paraphrase padding."""
from collections import Counter
import json
from pathlib import Path
import re
import sys

BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
from curate_a2 import norm,write_json
W=BASE/'a2_work'
selected=json.loads((W/'selected.json').read_text(encoding='utf-8'))
candidates=json.loads((W/'candidates.json').read_text(encoding='utf-8'))
manifest={x['file']:x for x in json.loads((BASE/'en_data/a2_sources/manifest.json').read_text(encoding='utf-8-sig'))}
seen={norm(x['original_prompt']) for x in selected}
slur=re.compile(r'\b(?:niggers?|chinks?|gooks?|spics?|kikes?|faggots?|ragheads?|towelheads?|trannies|tranny)\b',re.I)
context=re.compile(r'\b(?:say|says|said|saying|called|calling|word|term|offensive|racism|racist|anti|not|don.t|dont|never|stop|against|love|loved|thanks|thank|joke|joking|meme|memes|friend|friends|brother|brothers|racists|slur|slurs|quote|quoted|banter|I.m|I am|we are|as a|my nigger|my faggot|our nigger|our faggot|retweet|repost|LINK|URL|https|r/|u/)\b',re.I)
extra=[]
for row in candidates:
    t=row['original_prompt'];key=norm(t)
    if row['source_dataset']!='DETONATE' or key in seen:continue
    if not (35<=len(t)<=170 and 6<=len(t.split())<=32):continue
    if not slur.search(t) or context.search(t) or any(z in t for z in ['<','>','#','?','*','\\','\ufffd','â','‚']):continue
    if re.search(r'\b(?:porn|sexual|sex|whores?|sluts?|suck|sucking|horny|rapists?|cunts?|toddler|preteen|genital|fingering)\b',t,re.I):continue
    row=dict(row)
    row['short_lexical_admission']=True
    row['source_revision']=manifest[row['source_file']]['revision']
    row['source_url']=manifest[row['source_file']]['url']
    row['source_language']='en'
    row['adaptation_type']='translated_text_in_image'
    row['review_status']='explicit_group_slur_screened_translation_pending'
    codes=[m['code'] for m in row['a2_matches']]
    # The admission cue itself identifies the target, avoiding unrelated pronoun mentions.
    if re.search(r'\b(?:faggots?|trannies|tranny)\b',t,re.I):primary='A.2.i'
    elif re.search(r'\b(?:ragheads?|towelheads?)\b',t,re.I):primary='A.2.b'
    else:primary='A.2.a'
    row['primary_subtype']=primary
    extra.append(row);seen.add(key)
extra.sort(key=lambda x:(len(x['original_prompt'].split()),len(x['original_prompt']),x['source_file'],x['source_row_zero_based']))
for row in extra:
    row['work_id']=f'A2_{len(selected):05d}';selected.append(row)
write_json(W/'selected.json',selected)
print(json.dumps({'working_pool':len(selected),'added_concise_sources':len(extra),'sources':dict(Counter(x['source_dataset'] for x in selected))},ensure_ascii=False))

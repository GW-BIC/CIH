# -*- coding: utf-8 -*-

import re
import time
import traceback
import adodbapi
import random
import xlsxwriter

def cleanText(text):
        newtext = re.sub(r'\s+', ' ', text.lower())
        newtext = re.sub(r'[^a-z0-9]', ' ', newtext)
        newtext = re.sub(r'\b\d+\b', '', newtext)
        return newtext.strip()
def check_nonascii(text):
    nonascii = []
    for c in text:
        if ord(c)>127:
            nonascii.append(c)
    return nonascii

if __name__ == '__main__':
    mainDir2 = ''
    connectors = ()
    sql = '''
select top 120 a.TIUDocumentSID, b.ReportText
from Dflt.YS_TIU_%s as a
join Src.STIUNotes_TIUDocument_8925 as b
on a.TIUDocumentSID = b.TIUDocumentSID
order by a.RandID
'''
    cam = 'ACUP'
    ptn1 = re.compile(r'\b(?:acupuncture|acup)\b',re.I|re.DOTALL)
#    cam = 'BIOF'
#    ptn1 = re.compile(r'\bbiofeedback\b',re.I|re.DOTALL)
#    cam = 'GUID'
#    ptn1 = re.compile(r'\bguided\W{1,10}(?:imagery|visualization)\b',re.I|re.DOTALL)
#    cam = 'MEDI'
#    ptn1 = re.compile(r'\b(?:meditation|mindfulness|MBSR|MBCT|Mantram)\b',re.I|re.DOTALL)
#    cam = 'TAIC'
#    ptn1 = re.compile(r'\b(?:tai\W{0,5}chi|qi\W{0,5}gong)\b',re.I|re.DOTALL)
#    cam = 'YOGA'
#    ptn1 = re.compile(r'\b(?:yoga|breathing\W{1,5}stretching\W{1,5}relaxation|pranayama|vinyasa|hatha)\b',re.I|re.DOTALL)
    print(cam)
    ptn2 = re.compile(r'\S+(\s+\S+){0,30}')
    print('Connecting to database ...', end=' ')
    conn = adodbapi.connect(';'.join(connectors),600)
    db = conn.cursor()
    print('done')
    print(time.strftime('%H:%M:%S'))
    try:
        snippets = []
        print(sql %cam)
        db.execute(sql %cam)
        for row in db:
            docID = row['TIUDocumentSID']
            text = row['ReportText']
            if text is None:
                print(docID)
                continue
            text = text.replace('\r\n','\n')
            text = text.replace('\r','\n')
            kwpos = []
            for m in re.finditer(ptn1,text):
                wStart = m.start()
                wEnd = m.end()
                kwpos.append((m.start(),m.end()))
            segStart = segEnd = len(text)+1
            snipID = 0
            for i in range(len(kwpos)):
                wStart,wEnd = kwpos[-i-1]
                if wEnd>segStart:
                    continue
                snipID -= 1
                mm = re.search(ptn2,text[wStart::-1])
                segStart = wStart-mm.end()+1
                mm = re.search(ptn2,text[wEnd-1:])
                segEnd = wEnd+mm.end()-1
                snippet = text[segStart:segEnd]
                kw = text[wStart:wEnd]
                snippets.append((docID,snipID,wStart,wEnd,kw,segStart,segEnd,snippet))
        print(len(snippets))
        random.seed(1)
        idxs = list(range(len(snippets)))
        random.shuffle(idxs)
        db.execute(
'''CREATE TABLE Dflt.YS_TIU_%s_Snip120 (
RowID int, TIUDocumentSID bigint, SnipID int, KWStart int, KWEnd int,
KW varchar(50), SegStart int, SegEnd int, Snippet varchar(max))
''' %cam)
        for n,i in enumerate(idxs[:120],start=1):
            docID,snipID,wStart,wEnd,kw,segStart,segEnd,snippet = snippets[i]
            db.execute(
"INSERT INTO Dflt.YS_TIU_%s_Snip120 VALUES (%d,%d,%d,%d,%d,'%s',%d,%d,'%s')"
%(cam,n,docID,snipID,wStart,wEnd,' '.join(kw.split()),
segStart,segEnd,snippet.replace("'","''")))
        conn.commit()
    except Exception:
        traceback.print_exc()
    finally:
        conn.close()
    print()
    print(time.strftime('%H:%M:%S'))

    cam = 'MEDI'
    sql = '''
select *
from Dflt.YS_TIU_%s_Snip120
order by RowID
'''
    snippets = []
    conn = adodbapi.connect(';'.join(connectors),600) #timeout = 600sec = 10min
    try:
        db = conn.cursor()
        db.execute(sql %cam)
        for row in db:
            n,tiu,snipid,kwstart,kwend,kw,segstart,segend,snip = row
            snippets.append(row)
            nonascii = check_nonascii(snip)
            if nonascii:
                print(n,i,nonascii)
    except Exception:
        traceback.print_exc()
    finally:
        conn.close()

    book = xlsxwriter.Workbook(mainDir2+r'\Snippets_%s_120.xlsx' %cam)
    sheet = book.add_worksheet()
    headers = ('Snippet ID','Modality','Keyword','Snippet Text')
    for j,header in enumerate(headers):
        sheet.write(0,j,header)
    for i in range(len(snippets)):
        n,tiu,snipid,wstart,wend,kw,segstart,segend,snippet = snippets[i]
        items = ['[%d]-%s-%s' %(n,tiu,snipid), cam, kw, snippet]
        for j,item in enumerate(items):
            sheet.write(n,j,item)
    book.close()

    with open(mainDir2+r'\Snippets_%s_120.xlsx.vtt' %cam, encoding='utf-8') as f:
        lines = f.readlines()
    nLines = len(lines)
    textStart, textEnd, markUpStart = -1, -1, -1
    newlines = []
    idx = -1
    sepLines = []
    pos = 0
    for n in range(nLines):
        newlines.append(lines[n])
        if textStart>0 and textEnd<0 and lines[n].startswith('#<-----'):
            textEnd = n
            sepLines.append(n+1)
        if textStart==0 and lines[n].startswith('#<-----'):
            textStart = n+1
            sepLines.append(n)
        if textStart<0 and lines[n].startswith('#<Text Content>'):
            textStart = 0
        if textStart>0 and textEnd<0 and lines[n]=='-'*82+'\n':
            sepLines.append(n)
        if markUpStart>0:
            items = lines[n].split('|')
            subitems = items[4].split('<::>')
            snippetNumber = int(subitems[0].split('="')[1][:-1])
            columnName = subitems[2].split('="')[1][:-1]
            if columnName=='Snippet Text':
                idx += 1
                pos += len(''.join(lines[sepLines[idx]+1:sepLines[idx]+6]))
                #label,start,end,snippet = rawdata1[idx]#rawdata[objs[idx]]
                kwstart,kwend,kw,segstart,segend,snippet = snippets[idx][3:]
                start,end = kwstart-segstart,kwend-segstart
                label = 'Keyword'
                segment = snippet[start:end]
                newlines.append('%d|%d|%s|||%s\n'%(pos+start,end-start,label,segment.replace('\n',' ')))
                pos += len(''.join(lines[sepLines[idx]+6:sepLines[idx+1]+1]))
        if markUpStart==0 and lines[n].startswith('#<-----'):
                markUpStart = n+1
        if markUpStart<0 and lines[n].startswith('#<MarkUps Information>'):
                markUpStart = 0
    with open(mainDir2+r'\Snippets_%s_120.vtt' %cam,'w',encoding='utf-8') as f:
        for line in newlines:
            f.write(line)

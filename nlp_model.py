# -*- coding: utf-8 -*-

import xlrd
import re
import numpy as np
import adodbapi
from sklearn import svm, metrics
import random
from sklearn.model_selection import KFold

def ReadVTT(vttfile):
    metas,snippets,seglabels = [],[],[]
    markUpStart = -1
    n = -1
    with open(vttfile,encoding='utf-8') as f:
        lines = [line for line in f]
    nLines = len(lines)
    textStart, textEnd, markUpStart = -1, -1, -1
    sepLines,labels = [],[]
    pos = 0
    for n in range(nLines):
        if markUpStart<0:
            for c in lines[n]:
                if ord(c)>=128:
                    print('Line',n,'has non-ascii characters')
                    print(repr(lines[n]))
                    break
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
            if items[2].startswith('SnippetColumn'):
                subitems = items[4].split('"')
                snippetNumber = int(subitems[1])
                if len(metas)==snippetNumber-1:
                    metas.append({})
                if len(seglabels)==snippetNumber-1:
                    seglabels.append([])
                columnName = subitems[-2]
                columnValue = items[5].rstrip()
                if len(columnValue)>50:
                    columnValue = columnValue[:50]+'...'
                metas[snippetNumber-1][columnName] = columnValue
            else:
                label = items[2].strip()
                start = int(items[0])
                end = start+int(items[1])
                labels.append((label,start,end))
        if markUpStart==0 and lines[n].startswith('#<-----'):
            markUpStart = n+1
        if markUpStart<0 and lines[n].startswith('#<MarkUps Information>'):
            markUpStart = 0
    pos = 0
    j = 0
    for i in range(len(sepLines)-1):
        snippet = ''.join(lines[sepLines[i]+6:sepLines[i+1]-1])
        pos += len(''.join(lines[sepLines[i]+1:sepLines[i]+6]))
        snippets.append(snippet)
        sniplen = len(snippet)

        while j<len(labels) and labels[j][1]<=pos+sniplen:
            if pos<=labels[j][1]:
                #print(i,pos,pos+sniplen,j,labels[j][1])
                seglabels[i].append((labels[j][1]-pos,labels[j][2]-pos,labels[j][0]))
            j += 1
        pos += len(''.join(lines[sepLines[i]+6:sepLines[i+1]+1]))
    print(len(labels)-len(metas))
    return metas,snippets,seglabels

def featureExtract(snippet,kw,kwstart,kwend):
    ptn_pxp = re.compile(r' {0,2}[\[\(] *[xX\+] *[\]\)] {0,2}')
    ptn_pwp = re.compile(r' {0,2}[\[\(] *\-? *[\]\)] {0,2}')
    if True:
        textBef = snippet[:kwstart+1].lower()
        textBef = re.sub(ptn_pxp,' pxp ',textBef)
        textBef = re.sub(ptn_pwp,' pwp ',textBef)
        textBef = re.sub(r'\s+',' ',textBef)
        textBef = re.sub('[^a-z]',' ',textBef)
        wordsBef = textBef.split(' ')[:-1]
        wordBef2wt = {}
        words = [kw]+wordsBef[::-1]
        for i in range(1,len(words)):
            if i>30:
                break
            w = words[i]
            if w=='':
                continue
            wordBef2wt[w] = max(wordBef2wt.get(w,0),(31-i)/30.0)
            if words[i-1]=='':
                continue
            ww = w+'_'+words[i-1]
            wordBef2wt[ww] = max(wordBef2wt.get(ww,0),(31-i)/30.0)*1.2
        textAft = snippet[kwend-1:].lower()
        textAft = re.sub(ptn_pxp,' pxp ',textAft)
        textAft = re.sub(ptn_pwp,' pwp ',textAft)
        textAft = re.sub('\s+',' ',textAft)
        textAft = re.sub('[^a-z]',' ',textAft)
        wordsAft = textAft.split(' ')[1:]
        wordAft2wt = {}
        words = [kw]+wordsAft
        for i in range(1,len(words)):
            if i>30:
                break
            w = words[i]
            if w=='':
                continue
            wordAft2wt[w] = max(wordAft2wt.get(w,0),(31-i)/30.0)
            if words[i-1]=='':
                continue
            ww = words[i-1]+'_'+w
            wordAft2wt[ww] = max(wordAft2wt.get(ww,0),(31-i)/30.0)*1.2
        return wordBef2wt,wordAft2wt

if __name__ == '__main__':
    theDir = ''
    vttFiles = [
        r'\Acupuncture-Combined.vtt',
        r'\Biofeedback-Combined.vtt',
        r'\GuidedImagery-Combined.vtt',
        r'\Meditation-Combined.vtt',
        r'\TaiChi-Combined.vtt',
        r'\Yoga-Combined.vtt'
    ]
    ptns = [
        re.compile(r'\b(?:acupuncture|acup|needling)\b',re.I|re.DOTALL),
        re.compile(r'\b(?:biofeedback)\b',re.I|re.DOTALL),
        re.compile(r'\bguided.{1,10}(?:imagery|visualization)\b',re.I|re.DOTALL),
        re.compile(r'\b(?:meditation|mindfulness|MBSR|MBCT|Mantram|Mant)\b',re.I|re.DOTALL),
        re.compile(r'\b(?:t\'?ai.{0,5}chi|qi.{0,5}gong)\b',re.I|re.DOTALL),
        re.compile(r'\b(?:yoga|breathing.{1,10}stretching.{1,10}relaxation|pranayama|vinyasa|hatha)\b',re.I|re.DOTALL)
    ]
    snippets1 = []
    #allseglabels = []
    metas1 = []
    kws_pos1 = []
    mod_idxs1 = []
    labels1 = []
    book = xlrd.open_workbook(theDir+r'\Corrected\Corrected.xlsx')
    for m,vttfile in enumerate(vttFiles):
        sheet = book.sheet_by_index(m)
        for i in range(sheet.nrows):
            cells = sheet.row(i)
            labels1.append(str(cells[2].value))

        metas,snippets,seglabels = ReadVTT(theDir+r'\Combined'+vttfile)
        for i,snippet in enumerate(snippets):
            start = end = -1
            for s in re.finditer(ptns[m],snippet):
                start,end = s.start(),s.end()
            if start == -1:
                continue
            kw = snippet[start:end].upper()
            if m==2:
                kw = 'GUIDED IMAGERY'
            elif m==4:
                if kw[0]=='T':
                    kw = 'TAICHI'
                else:
                    kw = 'QIGONG'
            elif m==5:
                if kw[0]=='B':
                    kw = 'BSR'
            kws_pos1.append((kw,start,end))
            snippets1.append(snippets[i])
            metas1.append(metas[i])
            #allseglabels.append(seglabels[i])
            mod_idxs1.append((m,i))

    modrow2label = {}
    theDir = ''
    book = xlrd.open_workbook(theDir+r'\CIHSnippets-SecondReview-02.24.xlsx')
    sheet = book.sheet_by_index(0)
    for i in range(1,sheet.nrows):
        cells = sheet.row(i)
        mod,rowid,label = str(cells[0].value),int(cells[1].value),str(cells[9].value)
        modrow2label[mod+str(rowid)] = label

    connectors = ()
    sql = '''
select Modality, RowID, Snippet
from Dflt.TBI_60K_SnipSample
where RowID%2=0
order by Modality, RowID
'''
    s300kws_pos = []
    s300snippets = []
    s300labels = []
    mods = ['ACUP','BIOF','GUID','MEDI','TAIC','YOGA']
    mod2idx = {mod:i for i,mod in enumerate(mods)}
    print('Connecting to database ...', end=' ')
    conn = adodbapi.connect(';'.join(connectors),600)
    db = conn.cursor()
    print('Connected.')
    try:
        db.execute(sql)
        for row in db:
            mod,rowid,snippet = str(row[0]),str(row[1]),str(row[2])
            label = modrow2label[mod+rowid]
            s300snippets.append(snippet)
            s300labels.append(label)
            m = mod2idx[mod]
            start = end = -1
            for s in re.finditer(ptns[m],snippet):
                start,end = s.start(),s.end()
            if start == -1:
                print(m,i)
                continue
            kw = snippet[start:end].upper()
            if m==2:
                kw = 'GUIDED IMAGERY'
            elif m==4:
                if kw[0]=='T':
                    kw = 'TAICHI'
                else:
                    kw = 'QIGONG'
            elif m==5:
                if kw[0]=='B':
                    kw = 'BSR'

            s300kws_pos.append((kw,start,end))
    finally:
        conn.close()

    theDir = ''
    vttFiles = [
        r'\Snippets_ACUP_120_Done.vtt',
        r'\Snippets_BIOF_120_Done.vtt',
        r'\Snippets_GUID_120_Done.vtt',
        r'\Snippets_MEDI_120_Done.vtt',
        r'\Snippets_TAIC_120_Done.vtt',
        r'\Snippets_YOGA_120_Done.vtt'
    ]
    s720snippets = []
    s720seglabels = []
    s720metas = []
    s720kws_pos,s720labels = [],[]
    for m,vttfile in enumerate(vttFiles):
        print(m,vttfile)
        metas,snippets,seglabels = ReadVTT(theDir+vttfile)
        s720metas.extend(metas)
        s720snippets.extend(snippets)
        s720seglabels.extend(seglabels)
        for i in range(len(seglabels)):
            snippet = snippets[i]
            x = 0
            for start,end,label in seglabels[i]:
                if label=='Relevant Context':
                    continue
                if label=='Keyword':
                    print(i,label)
                x += 1
                s720labels.append(label)
                kw = snippet[start:end].upper()
                if m==2:
                    kw = 'GUIDED IMAGERY'
                elif m==4:
                    if kw[0]=='T':
                        kw = 'TAICHI'
                    else:
                        kw = 'QIGONG'
                elif m==5:
                    if kw[0]=='B':
                        kw = 'BSR'
                s720kws_pos.append((kw,start,end))
            if x != 1:
                print(x,i,seglabels[i])

    def convert(label):
        if label in ('RC','Uncertain'):
            return 'U'
        if label in ('NO','DN','Definitely No'):
            return 'N'
        if label == 'Definitely Yes':
            return 'Y'
        if label == 'Probably Yes':
            return 'PY'
        return label
    snippets2 = s720snippets #s300snippets +
    labels2 = [convert(x) for x in s720labels]#s300labels +
    kws_pos2 = s720kws_pos#s300kws_pos +

    label2count = {}
    for i,label in enumerate(labels2):
        label2count[label] = label2count.get(label,0) + 1
    for label in sorted(label2count):
        print('%s\t%d' %(label,label2count[label]))


    kwdoc_data1,clss1 = [],[]
    for i in range(len(snippets1)):
        snippet = snippets1[i]
        kw,kwstart,kwend = kws_pos1[i]
        wordBef2wt,wordAft2wt = featureExtract(snippet,kw,kwstart,kwend)
        kwdoc_data1.append((wordBef2wt,wordAft2wt,kw))
        clss1.append(int(labels1[i][0] in ('Y','P')))
    kwdoc_data2,clss2 = [],[]
    for i in range(len(snippets2)):
        snippet = snippets2[i]
        kw,kwstart,kwend = kws_pos2[i]
        wordBef2wt,wordAft2wt = featureExtract(snippet,kw,kwstart,kwend)
        kwdoc_data2.append((wordBef2wt,wordAft2wt,kw))
        clss2.append(int(labels2[i][0] in ('Y','P')))

    kw2df = {}
    for _,_,kw in kwdoc_data1:
        kw2df[kw] = kw2df.get(kw,0)+1
    kws = sorted(kw2df)
    kw2idx = {kw:i for i,kw in enumerate(kws)}
    print(kw2idx)

    random.seed(1)
    idxs = list(range(len(clss2)))
    model = svm.LinearSVC(C=0.01)
    c_div = [1.2,1.2]

    Y_tests,Y_scores = [],[]
    ten_folds = KFold(n_splits=10)

    random.shuffle(idxs)
    k = 0
    for train,test in ten_folds.split(clss2):
        k += 1
        wordBef2df,wordAft2df = {},{}
        c_wordBef2df,c_wordAft2df = [{},{}],[{},{}]
        c_count = [0,0]
        for i_train in train:
            i = idxs[i_train]
            c = clss2[i]
            c_count[c] += 1
            wordBef2wt,wordAft2wt,_ = kwdoc_data2[i]
            for w in wordBef2wt:
                wordBef2df[w] = wordBef2df.get(w,0)+1
                c_wordBef2df[c][w] = c_wordBef2df[c].get(w,0)+1
            for w in wordAft2wt:
                wordAft2df[w] = wordAft2df.get(w,0)+1
                c_wordAft2df[c][w] = c_wordAft2df[c].get(w,0)+1
        for i in range(len(clss1)):
            c = clss1[i]
            c_count[c] += 1
            wordBef2wt,wordAft2wt,kw = kwdoc_data1[i]
            for w in wordBef2wt:
                wordBef2df[w] = wordBef2df.get(w,0)+1
                c_wordBef2df[c][w] = c_wordBef2df[c].get(w,0)+1
            for w in wordAft2wt:
                wordAft2df[w] = wordAft2df.get(w,0)+1
                c_wordAft2df[c][w] = c_wordAft2df[c].get(w,0)+1
        n_samples = sum(c_count)
        c_ratio = [c_count[c]*1.0/n_samples for c in (0,1)]
        c_highBef = [{},{}]
        for w in wordBef2df:
            if wordBef2df[w]<4:
                continue
            for c in (0,1):
                r = c_wordBef2df[c].get(w,0)*1.0/wordBef2df[w]
                if (1-r)*c_div[c]<1-c_ratio[c]:
                    c_highBef[c][w] = r
        featuresBef = sorted(set(c_highBef[0])|set(c_highBef[1]))
        featBef2idx = {w:i for i,w in enumerate(featuresBef)}
        c_highAft = [{},{}]
        for w in wordAft2df:
            if wordAft2df[w]<4:
                continue
            for c in (0,1):
                r = c_wordAft2df[c].get(w,0)*1.0/wordAft2df[w]
                if (1-r)*c_div[c]<1-c_ratio[c]:
                    c_highAft[c][w] = r
        featuresAft = sorted(set(c_highAft[0])|set(c_highAft[1]))
        featAft2idx = {w:i for i,w in enumerate(featuresAft)}
        uncovered = 0
        c_uncovered = [0,0]
        X_train = np.zeros((n_samples,len(featuresBef)+len(featuresAft)+len(kws)))
        Y_train = np.zeros(n_samples,dtype=int)
        n = -1
        for i_train in train:
            n += 1
            i = idxs[i_train]
            Y_train[n] = clss2[i]
            wordBef2wt,wordAft2wt,kw = kwdoc_data2[i]
            for w in wordBef2wt:
                if not w in featBef2idx:
                    continue
                j = featBef2idx[w]
                X_train[n,j] = wordBef2wt[w]
            for w in wordAft2wt:
                if not w in featAft2idx:
                    continue
                j = len(featuresBef)+featAft2idx[w]
                X_train[n,j] = wordAft2wt[w]
            if kw in kw2idx:
                j = len(featuresBef)+len(featuresAft)+kw2idx[kw]
                X_train[n,j] = 1
        for i in range(len(clss1)):
            n += 1
            Y_train[n] = clss1[i]
            wordBef2wt,wordAft2wt,kw = kwdoc_data1[i]
            for w in wordBef2wt:
                if not w in featBef2idx:
                    continue
                j = featBef2idx[w]
                X_train[n,j] = wordBef2wt[w]
            for w in wordAft2wt:
                if not w in featAft2idx:
                    continue
                j = len(featuresBef)+featAft2idx[w]
                X_train[n,j] = wordAft2wt[w]
            if kw in kw2idx:
                j = len(featuresBef)+len(featuresAft)+kw2idx[kw]
                X_train[n,j] = 1
        X_test = np.zeros((len(test),len(featuresBef)+len(featuresAft)+len(kws)))
        Y_test = np.zeros(len(test),dtype=int)
        n = -1
        for i_test in test:
            n += 1
            i = idxs[i_test]
            Y_test[n] = clss2[i]
            wordBef2wt,wordAft2wt,kw = kwdoc_data2[i]
            for w in wordBef2wt:
                if not w in featBef2idx:
                    continue
                j = featBef2idx[w]
                X_test[n,j] = wordBef2wt[w]
            for w in wordAft2wt:
                if not w in featAft2idx:
                    continue
                j = len(featuresBef)+featAft2idx[w]
                X_test[n,j] = wordAft2wt[w]
            if kw in kw2idx:
                j = len(featuresBef)+len(featuresAft)+kw2idx[kw]
                X_test[n,j] = 1
        model.fit(X_train,Y_train)
        Y_score = model.decision_function(X_test)
        Y_scores.append(Y_score)
        Y_tests.append(np.array(Y_test))
    Y_score_final = np.concatenate(Y_scores)
    Y_test_final = np.concatenate(Y_tests)
    fprs,tprs,thresholds = metrics.roc_curve(Y_test_final,Y_score_final)
    auc = metrics.auc(fprs,tprs)
    print('AUC: %.3f' %(auc*100))
    npos,nneg = Y_test_final.sum(),(1-Y_test_final).sum()

    accs = (tprs*npos+(1-fprs)*nneg)/(npos+nneg)
    ppvs = tprs*npos/(tprs*npos+fprs*nneg+(tprs+fprs==0))
    fscores = (1+1)*ppvs*tprs/(1*ppvs+tprs+(ppvs+tprs==0))
    idxmax = accs.argmax()
    print('Sensitivity:\t%.3f' %tprs[idxmax])
    print('Specificity:\t%.3f' %(1-fprs[idxmax]))
    print('Precision:\t%.3f' %ppvs[idxmax])
    print('F-Score:\t%.3f' %fscores[idxmax])

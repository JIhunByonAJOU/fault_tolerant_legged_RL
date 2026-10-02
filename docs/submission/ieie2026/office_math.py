"""Convert a documented LaTeX subset to editable Office Math, without Word.

Unsupported MathML nodes fail explicitly. This is a document-build helper, not
a symbolic mathematics engine. Preview images never replace delivery equations.
"""
from copy import deepcopy
from lxml import etree as E
from latex2mathml.converter import convert

M = 'http://schemas.openxmlformats.org/officeDocument/2006/math'
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
A = 'http://schemas.openxmlformats.org/drawingml/2006/main'
A14 = 'http://schemas.microsoft.com/office/drawing/2010/main'

def node(name, *children, **attrs):
    out = E.Element('{%s}%s' % (M, name))
    for k, v in attrs.items(): out.set('{%s}%s' % (M, k), str(v))
    for child in children: out.append(child)
    return out

def math_run(text, *, powerpoint=False, size=18):
    out = node('r')
    out.append(node('rPr', node('nor')))
    if powerpoint:
        pr = E.SubElement(out[0], '{%s}rPr' % A, sz=str(int(size*100)), lang='ko-KR')
        fill = E.SubElement(pr, '{%s}solidFill' % A)
        E.SubElement(fill, '{%s}srgbClr' % A, val='222222')
        for family in ['latin','ea','cs']: E.SubElement(pr,'{%s}%s' % (A,family),typeface='맑은 고딕')
    else:
        pr = E.SubElement(out,'{%s}rPr' % W)
        fonts = E.SubElement(pr,'{%s}rFonts' % W)
        for family in ['ascii','hAnsi','eastAsia']: fonts.set('{%s}%s' % (W,family),'Cambria Math')
        E.SubElement(pr,'{%s}sz' % W).set('{%s}val' % W,str(int(size*2)))
    t = E.SubElement(out,'{%s}t' % M); t.text=text
    return out

def equation(latex, *, powerpoint=False, size=18):
    root=E.fromstring(convert(latex).encode())
    def recurse(el):
        tag=E.QName(el).localname
        children=list(el)
        if tag in ['math','mrow','mstyle','mpadded','semantics']:
            return [x for child in children for x in recurse(child)]
        if tag in ['mi','mn','mo','mtext','ms']:
            return [math_run(el.text or '',powerpoint=powerpoint,size=size)]
        if tag=='mspace': return [math_run(' ',powerpoint=powerpoint,size=size)]
        if tag in ['msub','msup','msubsup']:
            names={'msub':['e','sub'],'msup':['e','sup'],'msubsup':['e','sub','sup']}[tag]
            kind={'msub':'sSub','msup':'sSup','msubsup':'sSubSup'}[tag]
            return [node(kind,*[node(name,*recurse(child)) for name,child in zip(names,children)])]
        if tag=='mfrac': return [node('f',node('num',*recurse(children[0])),node('den',*recurse(children[1])))]
        if tag in ['msqrt','mroot']:
            content=children if tag=='msqrt' else [children[0]]
            pr=node('radPr',node('degHide',val='1' if tag=='msqrt' else '0'))
            degree=node('deg',*(recurse(children[1]) if tag=='mroot' else []))
            return [node('rad',pr,degree,node('e',*[x for child in content for x in recurse(child)]))]
        if tag=='mfenced':
            return [math_run(el.get('open','('),powerpoint=powerpoint,size=size),
                    *[x for child in children for x in recurse(child)],
                    math_run(el.get('close',')'),powerpoint=powerpoint,size=size)]
        if tag=='mover' and len(children)==2:
            mark=''.join(children[1].itertext())
            accents={'˙':'\u0307','·':'\u0307','^':'\u0302','ˆ':'\u0302',
                     '→':'\u20d7','¯':'\u0304','‾':'\u0304','˜':'\u0303'}
            if el.get('accent')=='true' or mark in accents or mark in ['\u0307','\u0302','\u0304']:
                char=accents.get(mark,mark)
                return [node('acc',node('accPr',node('chr',val=char)),node('e',*recurse(children[0])))]
        if tag in ['mover','munder','munderover']:
            names={'mover':['e','sup'],'munder':['e','sub'],'munderover':['e','sub','sup']}[tag]
            kind={'mover':'sSup','munder':'sSub','munderover':'sSubSup'}[tag]
            return [node(kind,*[node(name,*recurse(child)) for name,child in zip(names,children)])]
        if tag=='mtable':
            rows=[]
            for row in children:
                rows.append(node('mr',*[node('e',*recurse(cell)) for cell in row]))
            return [node('m',*rows)]
        if tag in ['mtr','mtd']: return [x for child in children for x in recurse(child)]
        raise ValueError('Unsupported MathML node: '+tag)
    return node('oMath',*recurse(root))

def put_word(paragraph, latex, size=9):
    paragraph._p.append(equation(latex,size=size))

def put_powerpoint(paragraph, latex, size=18, color='222222'):
    wrapper=E.Element('{%s}m'%A14,nsmap={'a14':A14,'m':M})
    wrapper.append(node('oMathPara',node('oMathParaPr',node('jc',val='left')),equation(latex,powerpoint=True,size=size)))
    for fill in wrapper.findall('.//{%s}srgbClr'%A):fill.set('val',color)
    paragraph._p.append(wrapper)

from typing import (
    List,
    Optional,
    NamedTuple,
    get_type_hints,
)

import enum

import os
import sys

import pickle

import zipfile
import wget

import re

import xml.sax
import xml.sax.xmlreader

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, '.data')

URL_UCD_ALL_FLAT_XML = r'https://www.unicode.org/Public/UCD/latest/ucdxml/ucd.all.flat.zip'

FN_UCD_ALL_FLAT_ZIP = os.path.join(DATA_DIR, 'ucd.all.flat.zip')
FN_UCD_ALL_FLAT_XML = os.path.join(DATA_DIR, 'ucd.all.flat.xml')
FN_UCD_PARSED = os.path.join(DATA_DIR, 'ucd.pickle')

_XML_Attrs = xml.sax.xmlreader.AttributesImpl


def ucd_age_int(s: str) -> int:
    if not s or s == "unassigned":
        return -1

    v = 0
    for i in s.split('.'):
        v = v * 1000 + int(i, base=10)
    return v


class ucd_bidi(NamedTuple):
    bc: Optional[str]
    Bidi_M: Optional[bool]
    bmg: Optional[str]
    Bidi_C: Optional[bool]
    bpt: Optional[str]  # bidi paired bracket type
    bpb: Optional[str]  # bidi paired bracket


class ucd_decomp(NamedTuple):
    dt: Optional[str]
    dm: Optional[str]
    CE: Optional[bool]
    Comp_Ex: Optional[bool]
    NFC_QC: Optional[str]
    NFD_QC: Optional[str]
    NFKC_QC: Optional[str]
    NFKD_QC: Optional[str]
    XO_NFC: Optional[bool]
    XO_NFD: Optional[bool]
    XO_NFKC: Optional[bool]
    XO_NFKD: Optional[bool]
    FC_NFKC: Optional[List[int]]


class ucd_numeric(NamedTuple):
    nt: Optional[str]
    nv: Optional[str]


class ucd_joining(NamedTuple):
    jt: Optional[str]
    jg: Optional[str]
    Join_C: Optional[bool]


class ucd_case(NamedTuple):
    Upper: Optional[bool]
    Lower: Optional[bool]
    OUpper: Optional[bool]
    OLower: Optional[bool]
    suc: Optional[int]
    slc: Optional[int]
    stc: Optional[int]
    uc: Optional[List[int]]
    lc: Optional[List[int]]
    tc: Optional[List[int]]
    scf: Optional[int]
    cf: Optional[List[int]]
    CI: Optional[bool]
    Cased: Optional[bool]
    CWCF: Optional[bool]
    CWCM: Optional[bool]
    CWL: Optional[bool]
    CWKCF: Optional[bool]
    CWT: Optional[bool]
    CWU: Optional[bool]
    NFKC_CF: Optional[List[int]]

    @property
    def is_simple(self):
        return (self.suc or self.slc or self.stc) and self.scf == self.slc

    @property
    def is_complex(self):
        return (self.uc or self.lc or self.tc)


class ucd_script(NamedTuple):
    sc: Optional[str]
    scx: Optional[List[str]]


class ucd_hangul(NamedTuple):
    hst: Optional[str]
    JSN: Optional[str]


class ucd_indic(NamedTuple):
    InSC: Optional[str]
    InMC: Optional[str]
    InPC: Optional[str]


class ucd_ID(NamedTuple):
    IDS: Optional[bool]
    OIDS: Optional[bool]
    XIDS: Optional[bool]
    IDC: Optional[bool]
    OIDC: Optional[bool]
    XIDC: Optional[bool]
    Pat_Syn: Optional[bool]
    Pat_WS: Optional[bool]


class ucd_emoji(NamedTuple):
    Emoji: bool
    EPres: bool
    EMod: bool
    EBase: bool
    EComp: bool
    ExPict: bool


class ucd_alias(NamedTuple):
    type: str
    alias: str


class ucd_char(NamedTuple):
    type: str
    cp: int  #
    age: str  # 4.4.1 Age property
    na: str  # 4.4.2 Name properties
    aliases: Optional[List[ucd_alias]]  # 4.4.3 Name aliases
    blk: str  # 4.4.4 Block
    gc: str  # 4.4.5 General Category
    ccc: str  # 4.4.6 Combining properties
    bidi: ucd_bidi  # 4.4.7 Bidirectionality properties
    decomp: ucd_decomp  # 4.4.8 Decomposition properties
    numeric: ucd_numeric  # 4.4.9 Numeric Properties
    joining: ucd_joining  # 4.4.10 Joining properties
    linebreak: Optional[str]  # 4.4.11 Linebreak properties
    east_asian_width: Optional[str]  # 4.4.12 East Asian Width property
    case_: ucd_case  # 4.4.13 Case properties
    script: ucd_script  # 4.4.14 Script properties
    isc: Optional[str]  # 4.4.15 ISO Comment properties
    hangul: ucd_hangul  # 4.4.16 Hangul properties
    indic: ucd_indic  # 4.4.17 Indic properties
    id: ucd_ID  # 4.4.18 Identifier and Pattern and programming language properties

    # ...?


def _bool(value: Optional[str], *args) -> Optional[bool]:
    '''bool'''
    return value == 'Y'


def _str(value: Optional[str], *args) -> Optional[str]:
    '''string'''
    return value


def _cp(value: Optional[str], *args) -> Optional[int]:
    '''codepoint'''
    try:
        return int(value, base=16)
    except:
        pass


def _cp_list(value: Optional[str], *args) -> Optional[List[int]]:
    '''codepoint list'''
    res = []
    for v in value.split(' '):
        vx = _cp(v)
        if vx:
            res.append(vx)
    return res if len(res) else None


def _xml_cp_range(attrs: _XML_Attrs):
    cp = attrs.get('cp')

    if cp:
        cp = int(cp, base=16)
        return range(cp, cp + 1)

    first = int(attrs.get('first-cp'), base=16)
    last = int(attrs.get('last-cp'), base=16)
    return range(first, last + 1)


def _get_type_parser(t):

    if t is bool or t is Optional[bool]:
        return _bool
    elif t is str or t is Optional[str]:
        return _str
    elif t is int or t is Optional[int]:
        return _cp
    elif t is List[int] or t is Optional[List[int]]:
        return _cp_list
    else:
        pass


def get_ucd_type_parsers_parser(T, parsers: dict, defaults: dict = {}):
    def parser(attr, attrs: _XML_Attrs) -> T:
        kv = {}
        for name, parser in parsers.items():
            if name in defaults:
                parser = defaults[name]

            if callable(parser):
                v = parser(attrs.get(name), attrs)
            else:
                v = parser

            kv[name] = v

        return T(**kv)

    return parser


def get_ucd_type_parsers(T):
    hints = get_type_hints(T)
    parsers = {}

    for name, type in hints.items():
        parser = _get_type_parser(type)

        if parser is None:
            try:
                _parsers = get_ucd_type_parsers(type)
                parser = get_ucd_type_parsers_parser(type, _parsers)
            except:
                parser = None

        parsers[name] = parser

    return parsers


ucd_char_parsers = get_ucd_type_parsers(ucd_char)


class UCD(NamedTuple):
    ages: List[str]
    blocks: List[str]
    categories: List[str]
    bidi_classes: List[str]
    scripts: List[str]
    linebreaks: List[str]

    version: Optional[str]
    chars: List[ucd_char]


class UCD_XML_SAX_Handler(xml.sax.ContentHandler):
    def __init__(self) -> None:
        super().__init__()

        # some helpers to parse xml
        self.elements = []
        self.group = None
        self.focused_chars = None

        # parsed content
        self.version: Optional[str] = None
        self.chars: List[ucd_char] = []

        self.ages = set()
        self.blocks = set()
        self.categories = set()
        self.bidi_classes = set()
        self.scripts = set()
        self.linebreaks = set()

    def get_ucd(self):
        return UCD(
            self.ages,
            self.blocks,
            self.categories,
            self.bidi_classes,
            self.scripts,
            self.linebreaks,
            self.version,
            self.chars,
        )

    def startDocument(self):
        return super().startDocument()

    def startElement(self, name: str, attrs: _XML_Attrs):
        self.elements.append(name)

        if name == 'group':
            self.group = attrs
        elif name in ('char', 'reserved', 'noncharacter', 'surrogate'):
            parser = get_ucd_type_parsers_parser(
                ucd_char,
                ucd_char_parsers,
                {
                    'type': name,
                    'cp': -1,
                    'aliases': []
                },
            )

            char = parser(None, attrs)

            first = len(self.chars)
            added = False

            for cp in _xml_cp_range(attrs):
                props = char._asdict()
                props['cp'] = cp

                char = ucd_char(**props)

                if not added:
                    added = True

                    self.blocks.add(char.blk)
                    self.ages.add(char.age)
                    self.categories.add(char.gc)
                    self.bidi_classes.add(char.bidi.bc)
                    self.scripts.add(char.script.sc)
                    self.linebreaks.add(char.linebreak)

                self.chars.append(char)

            last = len(self.chars)
            self.focused_chars = range(first, last)

        elif name in ('name-alias'):
            if self.focused_chars:
                for i in self.focused_chars:
                    self.chars[i].aliases.append(
                        ucd_alias(
                            attrs.get('type'),
                            attrs.get('alias'),
                        ))
        else:
            self.focused_chars = None

    def endElement(self, name):
        assert self.elements[-1] == name

        self.elements.pop(-1)

        if name == 'group':
            self.group = None
        elif name == 'repertoire':
            self.blocks = sorted(self.blocks)
            self.ages = sorted(self.ages, key=ucd_age_int)
            self.categories = sorted(self.categories)
            self.bidi_classes = sorted(self.bidi_classes)
            self.scripts = sorted(self.scripts)
            self.linebreaks = sorted(self.linebreaks)

    def characters(self, content):
        if self.elements[-1] == 'description':
            m = re.match(r'Unicode (\d+.\d+.\d+)', content)
            if m:
                self.version = m.group(1)
                print(f'Unicode {self.version}')


def get_inttype(*sequence):
    inttypes = [
        ('uint8_t', 1),
        ('uint16_t', 2),
        ('uint32_t', 4),
        ('int8_t', 1),
        ('int16_t', 2),
        ('int32_t', 4),
    ]
    limits = [
        (0, 255),
        (0, 65535),
        (0, 4294967295),
        (-128, 127),
        (-32768, 32767),
        (-2147483648, 2147483647),
    ]
    minval, maxval = min(*sequence), max(*sequence)
    for i, (minlimit, maxlimit) in enumerate(limits):
        if minlimit <= minval and maxval <= maxlimit:
            return inttypes[i]


def get_size(t):
    max_val = max(t)
    if max_val < 256:
        return 1
    elif max_val < 65536:
        return 2
    else:
        return 4


def _prepare():
    ''' prepare ucd xml file
    '''
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    if not os.path.exists(FN_UCD_ALL_FLAT_XML):
        sys.stdout.write('download ucd all flat xml file from the internet')
        fn = wget.download(URL_UCD_ALL_FLAT_XML, out=FN_UCD_ALL_FLAT_ZIP)
        sys.stdout.write('\r\n')

        with zipfile.ZipFile(fn, mode='r', allowZip64=True) as zipf:
            zipf.extract('ucd.all.flat.xml', DATA_DIR)


def _ucd() -> 'UCD':
    ''' get ucd
    '''
    try:
        if os.path.exists(FN_UCD_PARSED):
            with open(FN_UCD_PARSED, 'rb') as fp:
                return pickle.load(fp)
    except:
        pass

    handler = UCD_XML_SAX_Handler()

    xml.sax.parse(FN_UCD_ALL_FLAT_XML, handler)

    ucd = handler.get_ucd()

    with open(FN_UCD_PARSED, 'wb') as fp:
        pickle.dump(ucd, fp)

    return ucd


class UCD_Flags(enum.IntFlag):
    Empty = 0
    Alpha = 1 << 1
    Decimal = 1 << 2
    Digit = 1 << 3
    Numeric = 1 << 4
    Upper = 1 << 5
    Lower = 1 << 6
    Title = 1 << 7
    Linebreak = 1 << 8
    Space = 1 << 9

    Printable = 1 << 10
    Case_Ignorable = 1 << 11
    Cased = 1 << 12
    Extra_Case = 1 << 13

    @staticmethod
    def c_iterate():
        for f in UCD_Flags:
            yield (f.name, f.value)


class UCD_Prop(NamedTuple):
    upper: int
    lower: int
    title: int
    flags: UCD_Flags
    decimal: int
    digit: int

    @staticmethod
    def c_iterate(table: List['UCD_Prop']):
        for i, field in enumerate(get_type_hints(UCD_Prop).keys()):
            type = get_inttype(*tuple(map(lambda en: en[i], table)))[0]
            yield (type, field)


def get_ucd_props_indexes(ucd: UCD):
    dummy = UCD_Prop(0, 0, 0, UCD_Flags.Empty, 0, 0)

    max_cp = max(ucd.chars, key=lambda x: x.cp).cp

    table = [dummy]
    cache = {dummy: 0}
    index = [0] * (max_cp + 1)
    numerics = {}
    extra_cases = []

    def get_ult(flags, cp, case_: ucd_case):
        x = lambda v: v if v else []
        uc, lc, tc = x(case_.uc), x(case_.lc), x(case_.tc)
        cf = x(case_.cf)

        upper, lower, title = 0, 0, 0

        if (len(uc) == 1 or len(lc) == 1 or len(tc) == 1) and lc == cf:
            if not tc:
                tc = uc
            x = lambda v: v[0] if v else cp
            upper = x(uc) - cp
            lower = x(lc) - cp
            title = x(tc) - cp

        elif 0 != len(uc) + len(lc) + len(tc) or len(cf) != 0:
            flags |= UCD_Flags.Extra_Case

            upper = len(extra_cases) | (len(uc) << 24)
            extra_cases.extend(uc)

            lower = len(extra_cases) | (len(lc) << 24)
            extra_cases.extend(lc)
            if cf:
                lower |= len(cf) << 20
                extra_cases.extend(cf)

            if uc == tc:
                title = upper
            else:
                title = len(extra_cases) | (len(tc) << 24)
                extra_cases.extend(tc)
        return flags, upper, lower, title

    for char in ucd.chars:
        if char.type in ('reserved', 'noncharacter', 'surrogate'):
            continue

        flags = UCD_Flags.Empty

        if char.gc in ["Lm", "Lt", "Lu", "Ll", "Lo"]:
            flags |= UCD_Flags.Alpha

        if char.case_.Cased:
            flags |= UCD_Flags.Cased

        if char.case_.CI:
            flags |= UCD_Flags.Case_Ignorable

        if char.case_.Upper or char.case_.OUpper:
            flags |= UCD_Flags.Upper

        if char.case_.Lower or char.case_.OLower:
            flags |= UCD_Flags.Lower

        if char.gc == 'Lt':
            flags |= UCD_Flags.Title

        if char.gc == 'Zs' or char.bidi.bc in ("WS", "B", "S"):
            flags |= UCD_Flags.Space

        if char.linebreak or char.bidi.bc == 'B':
            flags |= UCD_Flags.Linebreak

        if char.cp == ord(' ') or char.gc[0] not in ('C', 'Z'):
            flags |= UCD_Flags.Printable

        nt, nv = char.numeric
        decimal, digit = 0, 0
        if nt == 'De':
            flags |= UCD_Flags.Decimal
            decimal = int(nv, base=10)
        elif nt == 'Di':
            flags |= UCD_Flags.Digit
            digit = int(nv, base=10)
        elif nt == 'Nu':
            flags |= UCD_Flags.Numeric
            numerics.setdefault(nv, []).append(char.cp)
        elif nt == 'None':
            pass
        else:
            print(f'unknown numeric (nt={nt}, nv={nv})', file=sys.stderr)

        flags, upper, lower, title = get_ult(flags, char.cp, char.case_)

        item = UCD_Prop(
            upper=upper,
            lower=lower,
            title=title,
            flags=flags,
            decimal=decimal,
            digit=digit,
        )

        i = cache.get(item)
        if i is None:
            cache[item] = i = len(table)
            table.append(item)
        try:
            index[char.cp] = i
        except:
            print(f'codepoint={char.cp}, {hex(char.cp)}')
            raise

    index1, index2, shift = split_2bin(index)
    return table, extra_cases, (index1, index2, shift), numerics


def split_2bin(table):
    n = len(table) - 1
    max_shift = 0
    if n > 0:
        while n >> 1:
            n >>= 1
            max_shift += 1
    del n
    bytes = sys.maxsize  # smallest total size so far
    t = tuple(table)
    for shift in range(max_shift + 1):
        t1, t2 = [], []
        size = 2**shift
        cache = {}
        for i in range(0, len(t), size):
            bin = t[i:i + size]
            index = cache.get(bin)
            if index is None:
                index = len(t2)
                cache[bin] = index
                t2.extend(bin)
            t1.append(index >> shift)
        b = len(t1) * get_size(t1) + len(t2) * get_size(t2)
        if b < bytes:
            best = t1, t2, shift
            bytes = b
    return best


echo_indent_ch = '\t'
echo_file = sys.stdout


def echo(msg, indent=None, file=None):
    if indent is None:
        indent = 0

    if isinstance(indent, int):
        indent = echo_indent_ch * indent

    assert isinstance(indent, str)

    if not file:
        file = echo_file

    return print(indent, msg, file=file, sep='')


def echo_ucd_props_header(
    fp,
    version: str,
    table,
    extra_cases,
    index1,
    index1_t,
    index2,
    index2_t,
    shift,
):
    global echo_file
    echo_file = fp

    major, minor, patch = tuple(version.split('.'))

    echo('#pragma once')
    echo('')
    echo('#include <stddef.h>')
    echo('#include <stdint.h>')
    echo('')
    echo('#ifdef __cplusplus')
    echo('extern "C" {')
    echo('#endif')
    echo('')
    echo(f'#define ucd_version       "{version}"')
    echo(f'#define ucd_version_major {major}')
    echo(f'#define ucd_version_minor {minor}')
    echo(f'#define ucd_version_patch {patch}')
    echo('')
    echo('enum ucd_prop_flags {')
    for name, value in UCD_Flags.c_iterate():
        echo(f'UCD_PROP_FLAG_{name.upper()} = {value},', indent=1)
    echo('};')
    echo('')
    echo('struct ucd_prop {')
    for type, field in UCD_Prop.c_iterate(table):
        echo(f'{type} {field};', indent=1)
    echo('};')
    echo('')
    echo(f'extern struct ucd_prop ucd_props[{len(table)}];')
    echo('')
    echo(f'extern uint32_t ucd_props_extra_cases[{len(extra_cases)}];')
    echo('')
    echo('enum {')
    echo(f'ucd_props_index_shift = {shift},', indent=1)
    echo(f'ucd_props_index_shift_n = 1 << ucd_props_index_shift,', indent=1)
    echo('};')
    echo('')
    echo(f'extern {index1_t} ucd_props_index1[{len(index1)}];')
    echo('')
    echo(f'extern {index2_t} ucd_props_index2[{len(index2)}];')
    echo('')
    echo(f'extern double ucd_to_numeric(uint32_t ch);')
    echo('')
    echo(
        '#define ucd_get_prop_index(ch) (ucd_props_index2[ucd_props_index1[((ch) / ucd_props_index_shift_n)] + ((ch) % ucd_props_index_shift_n)])'
    )
    echo('#define ucd_get_prop(ch) (&ucd_props[ucd_get_prop_index(ch)])')
    echo('')
    echo('#ifdef __cplusplus')
    echo('}')
    echo('#endif')


def echo_ucd_props_source(
    fp,
    h_fn,
    table,
    extra_cases,
    index1,
    index2,
    numeric,
):
    global echo_file
    echo_file = fp

    step = 32

    def m2s(t):
        return f'{t}'

    def tf(t: UCD_Prop):
        return f'{{ {", ".join(map(m2s, t))} }},'

    echo(f'#include "{h_fn}"')
    echo('')
    echo(f'struct ucd_prop ucd_props[{len(table)}] = {{')
    for entry in table:
        echo(tf(entry), indent=1)
    echo('};')
    echo('')
    echo(f'uint32_t ucd_props_extra_cases[{len(extra_cases)}] = {{')
    for i in range(0, len(extra_cases), step):
        echo(f'{", ".join(map(m2s, extra_cases[i:i+step]))},', indent=1)
    echo('};')
    echo('')
    echo(f'{index1_t} ucd_props_index1[{len(index1)}] = {{')
    for i in range(0, len(index1), step):
        echo(f'{", ".join(map(m2s, index1[i:i+step]))},', indent=1)
    echo('};')
    echo('')
    echo(f'{index2_t} ucd_props_index2[{len(index2)}] = {{')
    for i in range(0, len(index2), step):
        echo(f'{", ".join(map(m2s, index2[i:i+step]))},', indent=1)
    echo('};')
    echo('')
    echo(f'double ucd_to_numeric(uint32_t ch) {{')
    echo('switch (ch) {', indent=1)
    for num, codepoints in sorted(numeric.items()):
        num = '/'.join(map(lambda x: str(float(x)), num.split('/')))
        for codepoint in sorted(codepoints):
            echo(f'case 0x{codepoint:05X}:', indent=2)
        echo(f'return (double) {num};', indent=3)
    echo('}', indent=1)
    echo('return -1.0;', indent=1)
    echo('}')


def _prepare_args(args=None):
    import argparse as ap

    parser = ap.ArgumentParser()

    parser.add_argument(
        '--workpath',
        '--pwd',
        default=os.path.abspath('.'),
    )

    parser.add_argument(
        '--name',
        default='ucd_props',
    )

    return parser.parse_args(args)


if __name__ == '__main__':
    opt = _prepare_args()

    _prepare()

    ucd = _ucd()

    print(ucd.version)
    print(ucd.ages)
    print(ucd.categories)

    table, extra_cases, _index, numeric = get_ucd_props_indexes(ucd)
    index1, index2, shift = _index

    index1_t = get_inttype(*tuple(index1))[0]
    index2_t = get_inttype(*tuple(index2))[0]

    h_fn = f'{opt.name}.h'
    c_fn = f'{opt.name}.c'
    abs_h_fn = os.path.join(opt.workpath, h_fn)
    abs_c_fn = os.path.join(opt.workpath, c_fn)

    with open(abs_h_fn, 'w', encoding='utf-8') as fp:
        echo_ucd_props_header(
            fp,
            ucd.version,
            table,
            extra_cases,
            index1,
            index2_t,
            index2,
            index2_t,
            shift,
        )

    with open(abs_c_fn, 'w', encoding='utf-8') as fp:
        echo_ucd_props_source(
            fp,
            h_fn,
            table,
            extra_cases,
            index1,
            index2,
            numeric,
        )

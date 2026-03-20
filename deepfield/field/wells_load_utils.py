"""Wells utils."""
import re
import shlex
import numpy as np
import pandas as pd

from .well_segment import WellSegment
from .utils import get_multout_paths, get_single_path
from .parse_utils import (read_rsm, parse_perf_line, parse_control_line,
                          parse_history_line, read_ecl_bin, parse_eclipse_keyword,
                          read_array, read_header, read_vfp_body, VFP_HEADER_INFO)

from .tables import VFPTable


DEFAULTS = {'RAD': 0.1524, 'DIAM': 0.3048, 'SKIN': 0, 'MULT': 1, 'CLOSE': False,
            'MODE': 'OPEN', 'DIR': 'Z', 'GROUP': 'FIELD'}

MODE_CONTROL = ['PROD', 'INJE', 'STOP']

VALUE_CONTROL = ['BHPT', 'THPT', 'DRAW', 'ETAB', 'OPT', 'GPT', 'WPT', 'LPT', 'VPT',
                 'OIT', 'GIT', 'WIT',
                 'HOIL', 'HGAS', 'HWAT', 'HLIQ', 'HBHP', 'HTHP', 'HWEF',
                 'GOPT', 'GGPT', 'GWPT', 'GLPT',
                 'GGIT', 'GWIT',
                 'OIL', 'GAS', 'WAT', 'LIQ', 'BHP', 'THP', 'GOR', 'OGR', 'WCT', 'WOR', 'WGR',
                 'DREF'
                 ]

def load_rsm(wells, path, logger):
    """Load RSM well data from file."""
    logger.info("Start reading {}".format(path))
    rsm = read_rsm(path, logger)
    logger.info("Finish reading {}".format(path))
    if '_global' not in rsm:
        logger.warning("Empty RSM file {}".format(path))
        return wells
    if '_children' in rsm['_global']:
        del rsm['_global']['_children']
    df = pd.DataFrame({k: v['data'] for k, v in rsm['_global'].items()})
    dates = pd.to_datetime(df[['YEAR', 'MONTH', 'DAY']])
    welldata = {}
    for wellname, data in rsm.items():
        if wellname == '_global':
            continue
        if '_children' in data:
            del data['_children']
        wellname = wellname.strip(' \t\'\"').upper()
        wdf = pd.DataFrame({k: v['data'] * v['multiplyer'] for k, v in data.items()})
        wdf['DATE'] = dates
        wdf = wdf[['DATE'] + [col for col in wdf.columns if col != 'DATE']]
        welldata[wellname] = {'RESULTS': wdf.sort_values('DATE')}
    return wells.update(welldata)

def load_ecl_binary(wells, path_to_results, basename, logger=None, **kwargs):
    """Load results from UNSMRY file."""
    _ = kwargs

    smry_path_unifout = get_single_path(path_to_results, basename + '.UNSMRY', logger)
    smry_path_multout = get_multout_paths(path_to_results, basename, r'S\d+')
    if smry_path_unifout is None and smry_path_multout is None:
        return wells

    spec_path = get_single_path(path_to_results, basename + '.SMSPEC', logger)
    if spec_path is None:
        return wells

    def is_well_name(s):
        return re.match(r"[a-zA-Z0-9]", s) is not None

    if smry_path_unifout:
        smry_data_tmp = read_ecl_bin(smry_path_unifout, attrs=['PARAMS'],
                                     sequential=True, logger=logger)['PARAMS']
    elif smry_path_multout:
        smry_data_tmp = [read_ecl_bin(
            path, attrs=['PARAMS'],
            sequential=True, logger=logger)['PARAMS'][0] for path in smry_path_multout]
    else:
        raise ValueError('Neither `summary_path_unifout` or `summary_path_multout` is defined.')
    smry_data = np.stack(smry_data_tmp) # type: ignore

    spec_dict = read_ecl_bin(spec_path, attrs=['KEYWORDS', 'WGNAMES'],
                             sequential=False, logger=logger)
    kw = [w.strip() for w in spec_dict['KEYWORDS']]
    wellnames = [w.strip() for w in spec_dict['WGNAMES']]

    df = pd.DataFrame({k: smry_data[:, kw.index(k)].astype(int)
                       for k in ['DAY', 'MONTH', "YEAR"]})
    dates = pd.to_datetime(df.YEAR*10000 + df.MONTH*100 + df.DAY, format='%Y%m%d')

    welldata = {w: {'RESULTS': pd.DataFrame({'DATE': dates})}
                for w in np.unique(wellnames) if is_well_name(w)}
    for i, w in enumerate(wellnames):
        if w not in welldata:
            continue
        welldata[w]['RESULTS'][kw[i]] = smry_data[:, i]
    for v in welldata.values():
        v['RESULTS'].sort_values('DATE', inplace=True)
    wells.state.binary_attributes.append('RESULTS')
    return wells.update(welldata)

def load_group(wells, buffer, **kwargs):
    """Load groups. Note: optional keyword FRAC is not implemented."""
    _ = kwargs
    group = next(iter(buffer)).upper().split('FRAC')[0].split()
    group_name = group[1]
    if group_name == '1*':
        group_name = DEFAULTS['GROUP']
    try:
        group_node = wells[group_name]
    except KeyError:
        group_node = WellSegment(parent=wells.root, name=group_name, ntype="group")
    for well in group[2:]:
        try:
            node = wells[well]
            node.parent = group_node
        except KeyError:
            WellSegment(parent=group_node, name=well)
    return wells

def load_grouptree(wells, buffer, **kwargs):
    """Load grouptree."""
    _ = kwargs
    for line in buffer:
        if line.strip() == '/':
            return wells
        node, grp = shlex.split(line.split('/')[0])[:2]#re.sub("[\"\']", "", line).split('/')[0].strip().split()
        if grp == '1*':
            grp = DEFAULTS['GROUP']
        try:
            node = wells[node]
        except KeyError:
            node = WellSegment(parent=wells.root, name=node, ntype="group")
        try:
            grp = wells[grp]
        except KeyError:
            grp = WellSegment(parent=wells.root, name=grp, ntype="group")
        node.parent = grp
    return wells

def _load_control_table(wells, attribute, columns, column_types, has_date, buffer, meta, **kwargs):
    _ = kwargs
    if has_date:
        dates = meta['DATES']
        date = dates[-1] if not dates.empty else wells.field.start
    else:
        date = None
    df = parse_eclipse_keyword(buffer, columns, column_types, DEFAULTS, date)
    if not df.empty:
        welldata = {}
        for k, v in df.groupby('WELL'):
            tmp = k.split('*')
            if len(tmp) == 1:
                welldata[k] = {
                    attribute: v.reset_index(drop=True)
                }
            elif len(tmp) == 2 and not tmp[1]:
                well_names = [name for name in
                              wells.main_branches if name.startswith(tmp[0])]
                for name in well_names:
                    welldata[name] = {
                        attribute: v.reset_index(drop=True).assign(WELL=name)
                    }
            else:
                raise ValueError(f'Cound not parse well name "{k}"')
        wells.update(welldata, mode='a', ignore_index=True)
        wells.fill_na(attribute)
    return wells

def load_welspecs(wells, buffer, meta, **kwargs):
    """Partial load WELSPECS table."""
    columns = ['WELL', 'GROUP', 'I', 'J', 'DREF', 'PHASE', 'DRAINAGE_RADIUS']
    column_types = {
        'text': columns[1:2] + columns[5:6],
        'int': columns[2:4],
        'float': columns[4:5] + columns[6:]
    }
    attribute = 'WELSPECS'
    has_date = False
    return _load_control_table(wells, attribute, columns, column_types, has_date,
                               buffer, meta, **kwargs)

def load_welspecl(wells, buffer, meta, **kwargs):
    """Partial load WELSPECL table."""
    columns = ['WELL', 'GROUP', 'LGR', 'I', 'J', 'DREF', 'PHASE', 'DRAINAGE_RADIUS']
    column_types = {
        'text': columns[1:3] + columns[6:7],
        'int': columns[3:5],
        'float': columns[5:6] + columns[7:]
    }
    attribute = 'WELSPECS' #ignore LGR and create WELSPECS attribute
    has_date = False
    return _load_control_table(wells, attribute, columns, column_types, has_date,
                               buffer, meta, **kwargs)

def load_wconprod(wells, buffer, meta, **kwargs):
    """Partial load WCONPROD table."""
    columns = ['DATE', 'WELL', 'MODE', 'CONTROL',
               'OPT', 'WPT', 'GPT', 'SLPT', 'LPT', 'BHPT']
    column_types = {
        'text': columns[1:4],
        'float': columns[4:]
    }
    attribute = 'WCONPROD'
    has_date = True
    return _load_control_table(wells, attribute, columns, column_types,
                               has_date, buffer, meta, **kwargs)

def load_wconinje(wells, buffer, meta, **kwargs):
    """Partial load WCONINJE table."""
    columns = ['DATE', 'WELL', 'PHASE', 'MODE', 'CONTROL', 'SPIT', 'PIT', 'BHPT']
    column_types = {
        'text': columns[1:5],
        'float': columns[5:]
    }
    attribute = 'WCONINJE'
    has_date = True
    return _load_control_table(wells, attribute, columns, column_types,
                               has_date, buffer, meta, **kwargs)

def load_wefac(wells, buffer, meta, **kwargs):
    """Partial load WEFAC table."""
    columns = ['DATE', 'WELL', 'WEF']
    column_types = {
        'text': columns[1:2],
        'float': columns[2:]
    }
    attribute = 'WEFAC'
    has_date = True
    return _load_control_table(wells, attribute, columns, column_types,
                               has_date, buffer, meta, **kwargs)

def load_compdat(wells, buffer, meta, **kwargs):
    """Load COMPDAT table."""
    columns = ['DATE', 'WELL', 'I', 'J', 'K1', 'K2', 'MODE', 'Sat',
               'CF', 'DIAM', 'KH', 'SKIN', 'ND', 'DIR', 'Ro']
    column_types = {
        'text': columns[1:2] + columns[6:7] + columns[13:14],
        'int': columns[2:6],
        'float': columns[7:13] + columns[14:15]
    }
    attribute = 'COMPDAT'
    has_date = True
    return _load_control_table(wells, attribute, columns, column_types,
                               has_date, buffer, meta, **kwargs)

def load_compdatl(wells, buffer, meta, **kwargs):
    """Load COMPDATL table."""
    columns = ['DATE', 'WELL', 'LGR', 'I', 'J', 'K1', 'K2', 'MODE', 'Sat',
               'CF', 'DIAM', 'KH', 'SKIN', 'ND', 'DIR', 'Ro']
    column_types = {
        'text': columns[1:3] + columns[7:8] + columns[14:15],
        'int': columns[3:7],
        'float': columns[8:14] + columns[15:16]
    }
    attribute = 'COMPDATL'
    has_date = True
    return _load_control_table(wells, attribute, columns, column_types,
                               has_date, buffer, meta, **kwargs)

def load_comdatmd(wells, buffer, meta, **kwargs):
    """Load COMPDATMD table"""
    columns = ['DATE', 'WELL', 'BRANCH', 'MDU', 'MDL', 'MD_TVD', 'MODE', 'Sat', 'CF', 'DIAM',
               'KH', 'SKIN', 'ND', 'MULT', 'TYPE']
    column_types = {
        'text': columns[1:2] + columns[5:7] + columns[14:15],
        'int': columns[2:3],
        'float': columns[3:5] + columns[7:14]
    }
    attribute = 'COMPDATMD'
    has_date = 'TRUE'
    return _load_control_table(wells, attribute, columns, column_types,
                               has_date, buffer, meta, **kwargs)

def load_wfrac(wells, buffer, meta, **kwargs):
    """Load WFRAC table"""
    columns = ['DATE', 'WELL', 'I1', 'J1', 'K1', 'I2', 'J2', 'K2', 'AZIMUTH_ANGLE',
               'ZENITH_ANGLE', 'HALF_LENGTH', 'WIDTH', 'PROPPANT', 'FLOW_FUNCTION',
               'PHASE',]
    column_types = {
        'text': columns[1:2] + columns[12:15],
        'int': columns[2:8],
        'float': columns[8:12]
    }
    attribute = 'WFRAC'
    has_date = 'TRUE'
    return _load_control_table(wells, attribute, columns, column_types,
                               has_date, buffer, meta, **kwargs)

def load_wfracp(wells, buffer, meta, **kwargs):
    """Load WFRACP table"""
    columns = ['DATE', 'WELL', 'I1', 'J1', 'K1', 'I2', 'J2', 'K2', 'AZIMUTH_ANGLE',
               'ZENITH_ANGLE', 'L1', 'L2', 'H1', 'H2', 'WIDTH', 'PROPPANT', 'FLOW_FUNCTION',
               'PHASE',]
    column_types = {
        'text': columns[1:2] + columns[15:18],
        'int': columns[2:8],
        'float': columns[8:15]
    }
    attribute = 'WFRACP'
    has_date = 'TRUE'
    return _load_control_table(wells, attribute, columns, column_types,
                               has_date, buffer, meta, **kwargs)

def load_welopen(wells, buffer, meta, **kwargs):
    """Load WELOPEN keyword.

    WELOPEN specifies per-well OPEN/SHUT status and belongs to the most
    recently read SCHEDULE date (i.e. the last item in `meta['DATES']`).
    """
    columns = ['DATE', 'WELL', 'STATUS']
    column_types = {'text': columns[1:]}
    attribute = 'WELOPEN'
    has_date = True
    return _load_control_table(wells, attribute, columns, column_types,
                               has_date, buffer, meta, **kwargs)

def load_welltracks(wells, buffer, **kwargs):
    """Load welltracks while possible.

    Parameters
    ----------
    buffer : StringIteratorIO
        Buffer to get string from.

    Returns
    -------
    comp : Wells
        Wells component with loaded well data.
    """
    _ = kwargs
    welldata = {}
    while 1:
        track = get_single_welltrack(buffer)
        if track:
            welldata.update(track)
        else:
            return wells.update(welldata)

def get_single_welltrack(buffer):
    """Load single welltrack."""
    track = []
    try:
        line = next(buffer)
    except StopIteration:
        return {}
    line = ' '.join([word for word in line.split() if word.upper() not in ['WELLTRACK']])
    name = line.strip(' \t\'\"').upper()
    last_line = False
    for line in buffer:
        if '/' in line:
            line = line.split('/')[0]
            last_line = True
        try:
            p = np.array(line.split(maxsplit=4)[:4]).astype(float)
            assert len(p) == 4
            track.append(p)
        except (ValueError, AssertionError):
            buffer.prev()
            break
        if last_line:
            break
    if track:
        return {name: {"WELLTRACK": np.array(track)}}
    return {}

def load_events(wells, buffer, column_names, logger, **kwargs):
    """Load perforations and events from event table."""
    _ = kwargs
    column_names = [s.upper() for s in column_names]
    if column_names[0] != 'WELL':
        logger.info("Expected WELL in the first column, found {}.".format(column_names[0]))
        return wells
    if column_names[1].strip('\'\"') != "DD.MM.YYYY":
        logger.info("Expected 'DD.MM.YYYY' in the second column, found {}.".format(column_names[1]))
        return wells
    column_names[1] = 'DATE'

    df_perf = pd.DataFrame()
    df_evnt = pd.DataFrame()

    for line in buffer:
        if 'ENDE' in line or line.strip() == '/':
            break
        if 'PERF' in line.upper():
            df_perf = pd.concat([df_perf, parse_perf_line(line, column_names, DEFAULTS)])
        elif 'SQUE' in line.upper():
            continue
        else:
            df_evnt = pd.concat([df_evnt, parse_control_line(line, MODE_CONTROL, VALUE_CONTROL)])

    if not df_perf.empty:
        welldata = {k: {'PERF': v.reset_index(drop=True).sort_values('DATE')}
                    for k, v in df_perf.groupby('WELL')}
        wells.update(welldata, mode='a', ignore_index=True)

    if not df_evnt.empty:
        welldata = {k: {'EVENTS': v.reset_index(drop=True).sort_values('DATE')}
                    for k, v in df_evnt.groupby('WELL')}
        wells.update(welldata, mode='a', ignore_index=True)
    return wells

def load_history(wells, buffer, column_names, logger, **kwargs):
    """Load history rates."""
    _ = kwargs
    column_names = [s.upper() for s in column_names]
    if column_names[0] != 'WELL':
        logger.info("Expected WELL in a first column, found {}.".format(column_names[0]))
        return wells
    if column_names[1].strip('\'\"') != "DD.MM.YYYY":
        logger.info("Expected 'DD.MM.YYYY' in a second column, found {}.".format(column_names[1]))
        return wells
    column_names[1] = 'DATE'

    df = pd.DataFrame()

    for line in buffer:
        if 'ENDE' in line or 'ENDH' in line or line.strip() == '/':
            break
        df = df.append(parse_history_line(line, column_names))

    if not df.empty:
        welldata = {k: {'HISTORY': v.reset_index(drop=True)} for k, v in df.groupby('WELL')}
        wells.update(welldata, mode='a', ignore_index=True)
    return wells


def load_network(wells, buffer, **kwargs):
    """Load network."""
    _ = kwargs
    columns = ['NODMAX', 'NBRMAX', 'NBCMAX']
    df = pd.DataFrame(dict(zip(columns, read_array(buffer, dtype=int).tolist())), index=[0])
    welldata = {'FIELD': {'NETWORK': df}}
    wells.update(welldata, mode='a', ignore_index=True)
    return wells

def load_branprop(wells, buffer, **kwargs):
    """Load branprop."""
    _ = kwargs
    columns = ['DOWNODE', 'UPNODE', 'VFPTAB', 'ALQ-NODE', 'ALQ-DEN']
    defaults = [None, None, None, 0.0, 'NONE']
    wellsdata = {}
    for line in buffer:
        line = line.split('/')[0].strip()
        if not line:
            break

        vals = re.sub("[\"\']", "", line).split()

        for i, col in enumerate(columns):
            try:
                val = vals[i]
                if val in ('1*', ""):
                    raise ValueError()
            except IndexError as e:
                if defaults[i] is None:
                    raise ValueError(f'Default value not assumed for {col} in the table.') from e
                vals.append(defaults[i])
            except ValueError as e:
                if defaults[i] is None:
                    raise ValueError(f'Default value not assumed for {col} in the table.') from e
                vals[i] = defaults[i]

        val_dict = dict(zip(columns, vals))
        if val_dict['VFPTAB'].isdigit():
            val_dict['VFPTAB'] = int(val_dict['VFPTAB'])
        else:
            raise ValueError('VFP Table number should be a positive integer')
        try:
            val_dict['ALQ-NODE'] = float(val_dict['ALQ-NODE'])
            if val_dict['ALQ-NODE'] < 0:
                raise ValueError()
        except ValueError as e:
            raise ValueError('Artificial Lift Quantity (ALQ-NODE) should be a positive integer') from e

        if val_dict['ALQ-DEN'] in ['DENO', 'DENG']:
            val_dict['ALQ-NODE'] = 0.0
        if val_dict['ALQ-DEN'] == 'NONE':
            val_dict['ALQ-DEN'] = None

        try:
            upnode = wells[val_dict['UPNODE']]
            upnode.ntype = "group"
        except KeyError:
            upnode = WellSegment(parent=wells.root,
                                 name=val_dict['UPNODE'],
                                 ntype='group') # upper node always be a group
        try:
            downode = wells[val_dict['DOWNODE']]
        except KeyError:
            downode = WellSegment(parent=wells.root,
                                  name=val_dict['DOWNODE'],
                                  ntype="group")

        vfp = get_vfp(wells, val_dict['VFPTAB'])
        if vfp is not None:
            downode.parent = upnode
        wellsdata[downode.name] = {
            'VFP': vfp,
            'ALQ_NODE': val_dict['ALQ-NODE'],
            'ALQ_DEN': val_dict['ALQ-DEN'],
        }
    wells.update(wellsdata, mode='w', **kwargs)
    return wells

def get_vfp(wells, num):
    """Get VFP Table object by its number."""
    try:
        num = int(num)
    except Exception as e:
        raise ValueError("VFP Table number should be convertable to integer, {} given".format(num)) from e
    if num == 9999:
        return VFPTable(data=None, name="VFP9999", dct={})
    if num == 0:
        return None
    matched = list(filter((lambda vfp: vfp.number == num), getattr(wells.root, "VFPTABLES", [])))
    if len(matched) < 1:
        raise ValueError('No VFP with this number is found: {}. VFP 9999 is chosen instead'.format(num))
    return matched[0]

def _iter_nodes_with_vfp(wells):
    """Yield nodes that currently have a VFP reference."""
    for node in wells:
        if 'VFP' in node.attributes:
            yield node

def _upsert_vfp_table(wells, vfp_table):
    """Insert or replace VFP table by table number and refresh references."""
    existing = list(getattr(wells.root, "VFPTABLES", []))
    for i, old in enumerate(existing):
        if old.number == vfp_table.number:
            existing[i] = vfp_table
            for node in _iter_nodes_with_vfp(wells):
                if getattr(node, 'VFP', None) is old:
                    node.VFP = vfp_table
            wells.update({'FIELD': {'VFPTABLES': existing}}, mode='w')
            return wells
    wells.update({'FIELD': {'VFPTABLES': [vfp_table]}}, mode='a')
    return wells

def _skip_vfp_body(buffer, n_rec):
    """Consume VFP body tokens without creating heavy arrays."""
    nflo = n_rec[0]
    nrow = int(np.prod(n_rec[1:]))
    nind = len(n_rec[1:])
    remaining = (nind + nflo) * nrow
    for line in buffer:
        tokens = line.strip().strip('/').split()
        if not tokens:
            continue
        remaining -= len(tokens)
        if remaining <= 0:
            return

def load_nodeprop(wells, buffer, **kwargs):
    """Load nodeprop."""
    _ = kwargs
    columns = ['NODE', 'PRESS', 'CHOKE', 'GASLIFT', 'GROUP']
    defaults = [None, "1*", 'NO', 'NO', '1*']
    wellsdata = {}
    for line in buffer:
        if line.strip() == '/':
            wells.update(wellsdata, mode='a')
            return wells

        vals = re.sub("[\"\']", "", line).split('/')[0].strip().split()
        for i, col in enumerate(columns):
            try:
                val = vals[i]
                if val in ('1*', ""):
                    raise ValueError()
            except IndexError as e:
                _ = e
                if defaults[i] is None:
                    raise ValueError(f'Default value not assumed for {col} in the table.') from None
                vals.append(defaults[i])
            except ValueError as e:
                _ = e
                if defaults[i] is None:
                    raise ValueError(f'Default value not assumed for {col} in the table.') from None
                vals[i] = defaults[i]

        val_dict = dict(zip(columns, vals))

        if val_dict['NODE'] not in wells:
            raise KeyError(f"Node {val_dict['NODE']} not found among previously described nodes.")
        if val_dict['PRESS'] != "1*":
            try:
                val_dict['PRESS'] = float(val_dict['PRESS'])
            except (TypeError, ValueError) as e:
                _ = e
                raise ValueError(f"PRESS can't be converted into number : {val_dict['PRESS']}") from None

        val_dict['CHOKE'] = val_dict['CHOKE'].upper() == 'YES'
        try:
            node = wells[val_dict['NODE']]
        except KeyError:
            node = WellSegment(parent=wells.root, name=val_dict['NODE'], ntype='node')

        wellsdata[node.name] = {k.upper():v for k, v in val_dict.items() if k != 'NODE'}
    wells.update(wellsdata, mode='a')
    return wells

def load_vfp(wells, buffer, attr, **kwargs):
    """Load VFP table."""
    deferred = kwargs.pop('deferred', False)
    ind_vars = ['FLO', 'THP'] + ['WFR', 'GFR', 'ALQ'] * (attr == 'VFPPROD')
    vfp_dict = {'TYPE': attr}
    vfp_dict['HEADER'] = read_header(buffer, VFP_HEADER_INFO[attr+'HEADER'])
    n_rec = []
    for rec in ind_vars:
        vfp_dict[rec] = read_array(buffer, dtype=np.float32)
        n_rec.append(vfp_dict[rec].size)
    number = vfp_dict['HEADER'].get('NUMBER', 0)
    if 'VFPTABL' in wells.root.attributes:
        name = f"VFP{wells.root.vfptabl}"
    else:
        name = "VFPI"

    if deferred:
        _skip_vfp_body(buffer, n_rec)
        # build a lightweight table shell without invoking VFP dct-parsing branch
        placeholder = VFPTable(data=pd.DataFrame(), name=name)
        placeholder.number = number
        placeholder.vfptype = attr
        placeholder.meta = dict(vfp_dict['HEADER'])
        placeholder.records = {}
        placeholder.variables = {}
        placeholder.constants = {}
        _upsert_vfp_table(wells, placeholder)
        return wells

    vfp_dict['DATA'] = read_vfp_body(buffer, n_rec)
    _upsert_vfp_table(wells, VFPTable(dct=vfp_dict, name=name))
    return wells

def load_vfptabl(wells, buffer, **kwargs):
    """Load VFPTABL keyword."""
    default = 3
    mode = next(iter(buffer)).split("/")[0].strip()
    mode = int(mode) if mode.isdigit() else default
    wellsdata = {'FIELD': {'VFPTABL': mode}}
    wells.update(wellsdata, mode='a', **kwargs)
    return wells

def load_netbalan(wells, buffer, **kwargs):
    """Load NETBALAN keyword."""
    _ = kwargs
    columns = ['INT', 'PRESTOL', 'MAXITER', "CHOKTOL"]
    defaults = ["1*", 0.1, 20, 0.01]
    vals = next(iter(buffer)).split("/")[0].strip().split()
    for i in range(len(columns)):
        try:
            val = vals[i]
            if val in ('1*', ""):
                raise ValueError()
        except IndexError:
            vals.append(defaults[i])
        except ValueError:
            vals[i] = defaults[i]
    df = pd.DataFrame(dict(zip(columns, vals)), index=[0])
    welldata = {'FIELD': {'NETBALAN': df}}
    wells.update(welldata, mode='a', ignore_index=True)
    return wells

def load_nliqrem(wells, buffer, **kwargs):
    """Partial load WCONPROD table."""
    _ = kwargs
    clause = kwargs["clause"]
    columns = ['NODE', 'MAXREM', 'MAXRAT']
    defaults = ["1*", np.inf, 1]
    col_def = dict(zip(columns, defaults))
    df = pd.DataFrame(columns=columns)
    for line in buffer:
        if '/' not in line:
            break
        line = line.split('/')[0].strip()
        if not line:
            break
        vals = line.split()[:len(columns)]
        full = [None] * len(columns)
        shift = 0
        for i, v in enumerate(vals):
            if i + shift >= len(columns):
                break
            if '*' in v:
                shift += int(v.strip('*')) - 1
            else:
                full[i+shift] = v
        df = df.append(dict(zip(columns, full)), ignore_index=True)
    
    for k, v in col_def.items():
        if k in df:
            df[k] = df[k].fillna(v)
    if not df.empty:
        df["MAXRAT"] = df["MAXRAT"].astype(np.float32)
        welldata = {k: {clause: v} for k, v in df.groupby('NODE')}
        wells.update(welldata, mode='a', ignore_index=True)
    return wells

def load_weltarg(wells, buffer, **kwargs):
    """Load weltarg."""
    _ = kwargs
    columns = ['WELNAME', 'TARGET', 'VALUE']
    defaults = [None, None, None]
    wellsdata = {}
    for line in buffer:
        line = line.split('/')[0].strip()
        if not line:
            break
        vals = re.sub("[\"\']", "", line).split()

        for i, col in enumerate(columns):
            try:
                val = vals[i]
                if val in ('1*', ""):
                    raise ValueError()
            except IndexError as e:
                if defaults[i] is None:
                    raise ValueError(f'Default value not assumed for {col} in the table.') from e
                vals.append(defaults[i])
            except ValueError as e:
                if defaults[i] is None:
                    raise ValueError(f'Default value not assumed for {col} in the table.') from e
                vals[i] = defaults[i]

        val_dict = dict(zip(columns, vals))
        
        if val_dict['TARGET'] not in ("VFP", "LIFT"):
            # warnings.warn("TARGET value is not currently supported: {}. Ignoring the line".format(val_dict['TARGET']), UserWarning)
            continue
        
        if val_dict['TARGET'] == "VFP":
            if val_dict['VALUE'].isdigit() and int(val_dict['VALUE']) > 0:
                val_dict['VALUE'] = int(val_dict['VALUE'])
            else:
                raise ValueError('VFP Table number should be a positive integer. Given {}.'.format(val_dict['VALUE']))
        if val_dict['TARGET'] == "LIFT":
            try:
                val_dict['VALUE'] = float(val_dict['VALUE'])
            except Exception as e:
                raise ValueError('ALQ should be a float. Given {}.'.format(val_dict['VALUE'])) from e

        try:
            well = wells[val_dict['WELNAME']]
        except KeyError:
            # warnings.warn("Well name not found: {}. Creating the node.".format(val_dict['WELNAME']), UserWarning)
            well = WellSegment(parent=wells.root,
                               name=val_dict['WELNAME'],
                               ntype='well')
        
        if well.name not in wellsdata:
            wellsdata[well.name] = {"ntype": "well"}
        if val_dict['TARGET'] == "VFP":
            vfp = get_vfp(wells, val_dict['VALUE'])
            wellsdata[well.name]['VFP'] = vfp
        if val_dict['TARGET'] == "LIFT":
            wellsdata[well.name]['ALQ'] = val_dict['VALUE']
    wells.update(wellsdata, mode='w', **kwargs)
    return wells
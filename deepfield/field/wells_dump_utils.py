"""Dump tools."""
import re

import pandas as pd

PERF_VALUE_COLUMNS = ['RAD', 'DIAM', 'SKIN', 'MULT']

def write_perf(f, wells, defaults):
    """Write perforations to file."""
    dfs = []
    for node in wells:
        if 'PERF' in node.attributes:
            if node.perf.empty:
                continue
            df = node.perf.copy()
            df['WELL'] = node.name
            dfs.append(df)
    if dfs:
        df = pd.concat(dfs, sort=False)
    else:
        return

    f.write('EFORm WELL \'DD.MM.YYYY\' MDL MDU ' +
            ' '.join([c for c in df if c in PERF_VALUE_COLUMNS]) + '\n')
    f.write('ETAB\n')

    for c in df.columns:
        if c in defaults:
            df[c] = df[c].fillna(defaults[c])
    if df.isna().any().any():
        raise ValueError('Perforations contains nans.')

    df['PERF'] = 'PERF'
    df['BRANCH'] = df['WELL'].str.split(':').apply(lambda x: 'BRANCH {}'.format(':'.join(x[1:]))
                                                   if len(x) > 1 else '')
    df['WELL'] = df['WELL'].str.split(':').apply(lambda x: x[0])
    df['DATE'] = df['DATE'].dt.strftime('%d.%m.%Y')
    if 'CLOSE' in df:
        df['CLOSE'] = df['CLOSE'].apply(lambda x: 'CLOSE' if x else '')

    df = df[['WELL', 'DATE', 'PERF', 'MDL', 'MDU'] +
            [c for c in df if c in PERF_VALUE_COLUMNS] +
            ['BRANCH'] +
            (['CLOSE'] if 'CLOSE' in df else [])]

    f.write(df.to_string(header=False, index=False, index_names=False) + '\n')
    f.write('ENDE\n')

def expand_event_df(df, value_control_kw):
    """Add control keywords to columns."""
    order = []
    for col in df.columns:
        if col in value_control_kw:
            df[col + '_'] = col
            order.extend([col + '_', col])
        elif col == 'MODE':
            order = ['MODE'] + order
    order = ['WELL', 'DATE'] + order
    return df[order]

def write_events(f, wells, value_control_kw):
    """Write perforations to file."""
    dfs = []
    for node in wells:
        if 'EVENTS' in node.attributes:
            if node.events.empty:
                continue
            df = node.events.copy()
            df['WELL'] = node.name
            dfs.append(df)
    if dfs:
        df = pd.concat(dfs, sort=False)
    else:
        return

    f.write('EFORm WELL \'DD.MM.YYYY\'\n')
    f.write('ETAB\n')

    df = df[['WELL'] + [col for col in df if col != 'WELL']]

    df['DATE'] = df['DATE'].dt.strftime('%d.%m.%Y')

    if 'MODE' in df:
        for _, df_mode in df.groupby('MODE'):
            df_mode = df_mode.dropna(axis=1)
            df_mode = expand_event_df(df_mode, value_control_kw)
            f.write(df_mode.to_string(header=False, index=False, index_names=False) + '\n')
    else:
        df = df.dropna(axis=1)
        f.write(df_mode.to_string(header=False, index=False, index_names=False) + '\n')
    f.write('ENDE\n')

def _quote_deck_token(token):
    """Quote token if it contains characters that deck parsers typically
    treat specially (e.g. '-' in 'P-40').
    """
    if token is None:
        return ''
    s = str(token)
    if re.search(r'[^A-Za-z0-9_]', s):
        return f"'{s}'"
    return s


def write_network(f, wells):
    """Dump NETWORK keyword if loaded."""
    if 'NETWORK' not in wells.root.attributes:
        return
    df = getattr(wells.root, 'NETWORK', None)
    if df is None or df.empty:
        return
    row = df.iloc[0]
    f.write('NETWORK\n')
    f.write(f"{int(row['NODMAX'])} {int(row['NBRMAX'])} {int(row['NBCMAX'])} /\n")
    f.write('/\n\n')


def write_netbalan(f, wells):
    """Dump NETBALAN keyword if loaded."""
    if 'NETBALAN' not in wells.root.attributes:
        return
    df = getattr(wells.root, 'NETBALAN', None)
    if df is None or df.empty:
        return
    row = df.iloc[0]
    f.write('NETBALAN\n')
    f.write(f"{row['INT']} {row['PRESTOL']} {row['MAXITER']} {row['CHOKTOL']} /\n")
    f.write('/\n\n')


def write_branprop(f, wells):
    """Dump BRANPROP keyword from node-level attributes."""
    from anytree import PreOrderIter

    rows = []
    for node in PreOrderIter(wells.root):
        if node.is_root:
            continue
        if 'VFP' not in node.attributes and 'ALQ_NODE' not in node.attributes:
            continue
        down = _quote_deck_token(node.name)
        up = _quote_deck_token(node.parent.name) if getattr(node, 'parent', None) else 'FIELD'

        vfp = getattr(node, 'VFP', None)
        vfptab = 0 if vfp is None else int(vfp.number)

        alq_node = float(getattr(node, 'ALQ_NODE', 0.0))
        alq_den = getattr(node, 'ALQ_DEN', None)
        alq_den_out = 'NONE' if alq_den is None else alq_den

        # Compact form: if ALQ_NODE is 0 and ALQ_DEN is NONE => omit params 4-5.
        if abs(alq_node) < 1e-12 and (alq_den is None or str(alq_den_out).upper() == 'NONE'):
            rows.append(f"{down} {up} {vfptab} /")
        else:
            rows.append(f"{down} {up} {vfptab} {alq_node} {alq_den_out} /")

    if not rows:
        return

    f.write('BRANPROP\n')
    for r in rows:
        f.write(r + '\n')
    f.write('/\n\n')


def write_nodeprop(f, wells):
    """Dump NODEPROP keyword from node-level attributes."""
    from anytree import PreOrderIter

    rows = []
    for node in PreOrderIter(wells.root):
        if node.is_root:
            continue
        if 'PRESS' not in node.attributes:
            continue

        n = _quote_deck_token(node.name)
        press = getattr(node, 'PRESS', '1*')
        press_out = '1*' if press is None else press

        choke = getattr(node, 'CHOKE', False)
        choke_out = 'YES' if bool(choke) else 'NO'

        gaslift = getattr(node, 'GASLIFT', 'NO')
        gaslift_out = 'YES' if str(gaslift).upper() == 'YES' else 'NO'

        group = getattr(node, 'GROUP', '1*')
        group_out = group

        # Compact form for default gaslift/group:
        if gaslift_out == 'NO' and str(group_out) == '1*':
            # If CHOKE is also default NO and deck allows omission, keep compact:
            if choke_out == 'NO':
                rows.append(f"{n} {press_out} /")
            else:
                rows.append(f"{n} {press_out} {choke_out} /")
        else:
            rows.append(f"{n} {press_out} {choke_out} {gaslift_out} {group_out} /")

    # NODEPROP in the deck usually includes the FIELD terminal node (root).
    # If root has PRESS, add it at the end.
    if 'PRESS' in wells.root.attributes:
        n = _quote_deck_token(wells.root.name)
        press = getattr(wells.root, 'PRESS', '1*')
        press_out = '1*' if press is None else press
        rows.append(f"{n} {press_out} /")

    if not rows:
        return

    f.write('NODEPROP\n')
    for r in rows:
        f.write(r + '\n')
    f.write('/\n\n')


def write_schedule(f, wells, dates, start_date, **kwargs):
    """Write SCHEDULE file."""

    def write_group(df, date, attr):
        group = df.loc[df['DATE'] == date]
        if group.empty:
            return
        f.write('{}\n'.format(attr))
        group = group.drop('DATE', axis=1).fillna('1*')
        f.write(group.to_string(header=False, index=False, index_names=False) + '\n')
        f.write('/\n\n')

    _ = kwargs
    attributes = ['COMPDAT', 'COMPDATL', 'COMPDATMD', 'WCONPROD', 'WCONINJE',
                  'WEFAC', 'WFRAC', 'WFRACP', 'WELOPEN']
    data = {key: [] for key in attributes}

    for node in wells:
        for attr, val in data.items():
            if attr in node.attributes:
                val.append(getattr(node, attr))

    data = {attr: pd.concat(val, sort=False) if val else pd.DataFrame(columns=['DATE']) for attr, val in data.items()}

    for val in data.values():
        val['END_LINE'] = '/'

    for i, date in enumerate(dates):
        str_date = date.strftime('%d %b %Y').upper()
        if not (i == 0 and start_date.date() == date.date()):
            f.write('DATES\n{} /\n/\n\n'.format(str_date))

        for attr, val in data.items():
            write_group(val, date, attr)


def write_welspecs(f, wells):
    """Write WELSPECS to file."""
    dfs = []
    for node in wells:
        if 'WELSPECS' in node.attributes and not node.welspecs.empty:
            welspecs = node.welspecs.copy()
            welspecs.loc[welspecs['GROUP']=='FIELD', 'GROUP'] = None
            dfs.append(welspecs)
    if not dfs:
        return

    df = pd.concat(dfs, sort=False).sort_values('WELL')
    df['END_LINE'] = '/'
    df.loc[pd.notnull(df[['I', 'J']]).values.any(axis=1), ['I', 'J']] = df.loc[
        pd.notnull(df[['I', 'J']]).values.any(axis=1), ['I', 'J']].astype(int)
    df = df.fillna('1*')

    f.write('WELSPECS\n')
    f.write(df.to_string(header=False, index=False, index_names=False) + '\n')
    f.write('/\n\n')

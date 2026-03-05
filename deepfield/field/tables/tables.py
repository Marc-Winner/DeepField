"""Tables component."""
import numpy as np
import pandas as pd
import h5py

from ..base_component import BaseComponent
from ..decorators import apply_to_each_input
from ..parse_utils import read_table, TABLE_INFO
from .table_interpolation import TABLE_INTERPOLATOR
from ..plot_utils import plot_table_1d, plot_table_2d


class Tables(BaseComponent):
    """Tables component of geological model."""

    @apply_to_each_input
    def apply(self, func, attr, *args, inplace=False, **kwargs):
        raise NotImplementedError()

    def _read_buffer(self, buffer, attr, **kwargs):
        """Read table data from string buffer.

        Parameters
        ----------
        buffer : buffer
            String buffer to read from.
        attr : str
            Target attribute.

        Returns
        -------
        comp : Tables
            Tables with new attribute.
        """
        dtype = kwargs.get('dtype', None)
        table = read_table(buffer, TABLE_INFO[attr], dtype, units=self.field.meta['UNITS'])
        setattr(self, attr, _Table(data=table, name=attr))
        return self

    def _load_hdf5(self, path, attrs=None, raise_errors=False, logger=None, **kwargs):
        """Load tables from HDF5 file.

        Parameters
        ----------
        path : str
            Path to file to load data from.
        attrs : str or array of str, optional
            Table names to get from file. If not given, loads all.

        Returns
        -------
        comp : Tables
            Tables with loaded attributes.
        """
        with h5py.File(path, 'r') as f:
            grp = f[self.class_name]
            for attr in grp.keys() if attrs is None else attrs:
                try:
                    table = pd.read_hdf(path, key='/'.join([grp.name, attr]), mode='r')
                except KeyError as err:
                    if raise_errors:
                        raise err
                    if logger is not None:
                        logger.info('Attribute %s not found in %s.' % (attr.upper(), grp.name))
                    continue
                setattr(self, attr, _Table(data=table, name=attr))
        return self

    def _dump_hdf5(self, path, mode='w', state=False, **kwargs):
        """Save tables into HDF5 file.

        Parameters
        ----------
        path : str
            Path to output file.
        mode : str
            Mode to open file.
            'w': write, a new file is created (an existing file with
            the same name would be deleted).
            'a': append, an existing file is opened for reading and writing,
            and if the file does not exist it is created.
            Default to 'a'.
        state : bool
            Dump compoments's state.

        Returns
        -------
        comp : Tables
            Tables unchanged.
        """
        _ = kwargs
        with h5py.File(path, mode) as f:
            tab = f[self.class_name] if self.class_name in f else f.create_group(self.class_name)
            if state:
                for k, v in self.state.as_dict().items():
                    tab.attrs[k] = v
        for attr, table in self.items():
            pd.DataFrame(table).to_hdf(path, key='/'.join([self.class_name, attr]), mode='a')
        return self

    def _dump_ascii(self, path, attrs=None, mode='w', **kwargs):
        """Save tables into ASCII file.

        Parameters
        ----------
        path : str
           Path to output file.
        attrs : str, array of str or None
           Array of keywords to dump into file.
        mode : str
           Mode to open file.
           'w': write, a new file is created (an existing file with
           the same name would be deleted).
           'a': append, an existing file is opened for reading and writing,
           and if the file does not exist it is created.
           Default to 'w'.
        fmt : str or sequence of strs, optional
           Format to be passed into ``numpy.savetxt`` function. Default to '%f'.
        Returns
        -------
        out : Tables
           Tables unchanged.
        """
        _ = kwargs
        if attrs is None:
            attrs = self.attributes
        elif isinstance(attrs, str):
            attrs = [attrs]
        with open(path, mode + 'b') as f:
            for att in attrs:
                table = getattr(self, att)
                table.dump_ascii(f)
        return self

    def pvtg_to_pvdg(self, as_saturated=False, inplace=True):
        """Transforms PVTG (wet gas) table into PVDG (dry gas) form.

        Parameters
        ----------
        as_saturated: bool
            If True, properties of the resulting dry gas will be similar to the properties of given saturated gas.
            Else, properties will be similar to the properties of given dry gas.
        inplace: bool
            If True, replaces PVTG table with a PVDG one. Else, returns PVDG as output.

        Returns
        -------
        pvdg: _Table or None
        """
        pvtg = self.pvtg

        pressure = pvtg.index.get_level_values('PRESSURE')
        rv = pvtg.index.get_level_values('RV')

        if as_saturated:
            mask = np.zeros_like(pressure, dtype=bool)
            for p in sorted(list(set(pressure))):
                level_rv = rv[pressure == p]
                sat_rv = np.max(level_rv)
                level_mask = np.all([pressure == p, rv == sat_rv], axis=0)
                mask = np.any([mask, level_mask], axis=0)
        else:
            mask = rv == 0

        pvdg = pvtg.loc[mask]
        pvdg.index = pvdg.index.get_level_values('PRESSURE')
        pvdg = _Table(data=pvdg, name='PVDG')
        if inplace:
            delattr(self, 'PVTG')
            setattr(self, 'PVDG', pvdg)
            return self
        return pvdg

    def pvt_oil(self, pressure, rs=None):
        """Caclulate FVF and Viscosity for oil."""
        if 'PVCDO' in self.attributes:
            return self.pvcdo(pressure)
        if 'PVDO' in self.attributes:
            return self.pvdo(pressure)
        return self.pvto(np.vstack((rs, pressure)).T)

class _Table(pd.DataFrame):  # pylint: disable=abstract-method
    """Table component."""
    _metadata = ['domain', 'name', '_interpolator']

    def __init__(self, data=None, **kwargs):
        self.name = kwargs.pop('name') if 'name' in kwargs else ''
        super().__init__(data=data, **kwargs)
        self.domain = list(self.index.names) if list(self.index.names)[0] is not None else None
        self._interpolator = None

    def __call__(self, x):
        """
        Apply table-defined function to x
        Parameters
        ----------
        x: array-like of shape (n_points, len(table.domain))
            Points for function to be computed at

        Returns
        -------
        values: array-like of shape (n_points, len(table.columns))
        """
        if self._interpolator is None:
            if self.name in TABLE_INTERPOLATOR:
                self._interpolator = TABLE_INTERPOLATOR[self.name](self)
            else:
                self._interpolator = TABLE_INTERPOLATOR[None](self)
        return self._interpolator(x)

    @property
    def _constructor(self):
        return self.__class__

    def plot(self, figsize=None):
        """Plot table."""
        if self.domain:
            if len(self.domain) == 1:
                plot_table_1d(self, figsize=figsize)
            elif len(self.domain) == 2:
                plot_table_2d(self, figsize=figsize)
            else:
                raise AttributeError('Can plot functions of 1 and 2 variables. Function of %d variables is given'
                                     % len(self.domain))
        else:
            raise AttributeError('The table has no domain. Hence, can not be plotted!')

    def dump_ascii(self, path_or_buffer):
        """Dumps table to ASCII format."""
        if self.domain is not None:
            header = self.name + '\n-- ' + '\t'.join(self.domain) + '\t' + '\t'.join(list(self.columns))
        else:
            header = self.name + '\n-- ' + '\t'.join(list(self.columns))
        footer = '/\n'
        round_decimals = 6

        if self.domain is not None:
            if len(self.domain) > 1:
                outer_idx = None
                idx_values = []
                row_ends = []
                for idx in self.index.values:
                    idx = [str(round(i, round_decimals)) for i in idx]
                    if idx[0] == outer_idx:
                        idx[0] = '\t'
                        row_ends.append(0)
                    else:
                        outer_idx = idx[0]
                        row_ends.append(1)
                    idx_values.append(idx)
                idx_values = np.array(idx_values)
                row_ends = np.array(row_ends + [1])[1:].astype(bool)
            else:
                idx_values = self.index.values.reshape(-1, 1)
                row_ends = np.zeros(self.shape[0]).astype(bool)
                row_ends[-1] = 1

            x = np.hstack([idx_values, np.round(self.values, round_decimals)]).astype(str)
        else:
            row_ends = np.zeros(self.shape[0]).astype(bool)
            row_ends[-1] = 1
            x = np.round(self.values, round_decimals).astype(str)

        for i in range(x.shape[0]):
            if row_ends[i]:
                x[i, -1] += '\t/'
        np.savetxt(path_or_buffer, x, header=header, footer=footer, delimiter='\t', comments='', fmt='%.18s')
        return self

    def to_numpy(self, include_index=False):
        """
        Get numpy representation of a table.
        """
        if include_index:
            if isinstance(self.index, pd.MultiIndex):
                index = np.array(self.index.values.tolist())
            else:
                index = self.index.values.reshape(-1, 1)
            return np.hstack((index, self.values))
        return self.values

class VFPTable(_Table):#pylint: disable=too-many-ancestors
    """VFPTable class"""

    _metadata = _Table._metadata + ['number', 'vfptype', 'records', 'variables',
                                    'constants', 'meta', '_model', '_finterpolator']

    def __init__(self, data=None, dct=None, name='VFPI', **kwargs):

        _empty = data not in [None, False]
        self.number = 9999
        if not _empty:
            dct = {} if dct is None else dct
            self.number = 9999 if name == 'VFP9999' else dct['HEADER'].pop('NUMBER', 0)
            self.vfptype = dct.get('TYPE', 'VFPPROD')
            self.meta = dct.get('HEADER', {})
            record_names = ['FLO', 'THP'] + ['WFR', 'GFR', 'ALQ'] * (self.vfptype == 'VFPPROD')
            self.records = {rec:dct.get(rec, None) for rec in record_names}
            self.variables = {'THP': record_names.index('THP')}
            self.constants = {rec: (i, None) for i, rec in enumerate(record_names) if rec not in self.variables}
            if self.number != 9999:
                kwargs['columns'] = list(self.records.keys())+[self.meta['VALUE']]
                data = dct['DATA']
            del dct

        super().__init__(data=data, name=name, **kwargs)

        if not _empty:
            self.format_table()
            if self.number == 9999:
                self._interpolator = lambda x: np.take(x, self.variables['THP'], axis=-1)
                self._finterpolator = None # fallback interpolator
            else:
                self._interpolator = TABLE_INTERPOLATOR[self.name](self)
                self._finterpolator = TABLE_INTERPOLATOR['VFPIE'](self)

    @property
    def _constructor(self):
        return VFPTable

    def __call__(self, x: np.ndarray) -> np.ndarray:

        x = np.asarray(x)
        try:
            out = self._interpolator(x)
            if np.any(np.isnan(out)):
                x_input = np.take(x, indices=list(self.variables.values()), axis=-1)
                out = self._finterpolator(x_input)
            return out
        except Exception as e:
            _ = e
            if x.shape[-1] == len(self.records):
                for const in self.constants:
                    index, value = self.constants[const]
                    arg = np.take(x, index, axis=-1)
                    if not np.allclose(arg, value):
                        raise ValueError(f"Arg {const} at {index} ({value}) doesn't match VFPTable ({arg})") from e
                x_input = np.take(x, list(self.variables.values()), axis=-1)
                out = self._interpolator(x_input)
                if np.any(np.isnan(out)):
                    out = self._finterpolator(x_input)
                return out
            raise ValueError(f"VFP {self.number}: col num != {len(self.variables)} or {len(self.records)}") from e

    @property
    def proper_records(self) -> dict[str, float]:
        """Return dict of records with length >1."""
        records = {k:v for k, v in self.records.items() if k in self.variables}
        return records

    @property
    def multi_grid(self) -> np.ndarray:
        """Return VFPTable reshaped into dimensions (rec1.size, ..., recn.size)."""
        nrec = [len(r) for r in self.proper_records.values()]
        vfp_array = self.sort_index(level=self.index.names[::-1]).values
        return vfp_array.reshape(nrec, order='F')

    def format_table(self) -> None:
        """Format table into appropriate format."""
        if self.number == 9999:
            return # no table formatting is required
        index2value = lambda index, rec: self.records[rec][int(index)] #pylint: disable=unnecessary-lambda-assignment
        self.variables: dict[str, int] = {}
        self.constants: dict[str, tuple[int, str]] = {}
        for i, rec in enumerate(self.records):
            self[rec] = self[rec].apply(index2value, args=(rec,))
            if self[rec].nunique() == 1:
                self.constants[rec] = (i, self.records[rec][0])
            else:
                self.variables[rec] = i
        self.drop(columns=list(self.constants.keys()), inplace=True)
        self.domain = list(self.variables.keys())
        self.set_index(self.domain, inplace=True)
"""Wells components."""
from functools import partial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy.optimize import minimize

from anytree import PreOrderIter, PostOrderIter, LevelOrderIter

from .parse_utils.ascii import INT_NAN
from .well_segment import WellSegment
from .base_tree import BaseTree
from .rates import show_rates, show_blocks_dynamics
from .grids import OrthogonalGrid
from .getting_wellblocks import get_wellblocks_vtk, get_wellblocks_compdat
from .wells_dump_utils import (
    write_perf, write_events, write_schedule, write_welspecs,
    write_network, write_netbalan, write_branprop, write_nodeprop,
)
from .wells_load_utils import (load_rsm, load_ecl_binary, load_group, load_grouptree,
                               load_welspecs, load_welspecl, load_compdat, load_compdatl,
                               load_comdatmd, load_wconprod, load_wconinje, load_welltracks,
                               load_events, load_history, load_wefac, load_wfrac, load_wfracp,
                               load_netbalan, load_network, load_branprop, load_nodeprop, 
                               load_vfp, load_vfptabl, load_nliqrem, load_weltarg,
                               load_welopen,
                               DEFAULTS, VALUE_CONTROL)
from .decorators import apply_to_each_segment, apply_to_each_node


class Wells(BaseTree):
    """Wells component.

    Contains wells and groups in a single tree structure, wells attributes
    and preprocessing actions.

    Parameters
    ----------
    node : WellSegment, optional
        Root node for well's tree.
    """

    def __init__(self, node=None, **kwargs):
        super().__init__(node=node, nodeclass=WellSegment, **kwargs)

    @property
    def main_branches(self):
        """List of main branches names."""
        return [node.name for node in self if node.is_main_branch]

    @property
    def event_dates(self):
        """List of dates with any event in main branches."""
        return self._collect_dates('EVENTS')

    @property
    def result_dates(self):
        """List of dates with any result in main branches."""
        return self._collect_dates('RESULTS')

    @property
    def history_dates(self):
        """List of dates with any history in main branches."""
        return self._collect_dates('HISTORY')

    def _collect_dates(self, attr):
        """List of common dates given in the attribute of main branches."""
        agg = [getattr(node, attr).DATE for node in self if node and attr in node]
        if not agg:
            return pd.to_datetime([])
        dates = sorted(pd.concat(agg).unique())
        return pd.to_datetime(dates)

    @property
    def total_rates(self):
        """Total rates over all wells."""
        return self.root.total_rates

    @property
    def cum_rates(self):
        """Cumulative rates over all wells."""
        return self.root.cum_rates

    def _get_fmt_loader(self, fmt):
        """Get loader for given file format."""
        if fmt == 'RSM':
            return self._load_rsm
        return super()._get_fmt_loader(fmt)

    def update(self, data, mode='w', **kwargs):
        """Update tree nodes with new wellsdata. If node does not exists,
        it will be attached to root.

        Parameters
        ----------
        data : dict
            Keys are well names, values are dicts with well attributes.
        mode : str, optional
            If 'w', write new data. If 'a', try to append new data. Default to 'w'.
        kwargs : misc
            Any additional named arguments to append.

        Returns
        -------
        out : Wells
            Wells with updated attributes.
        """
        def _get_parent(name, data):
            if ':' in name:
                return self[':'.join(name.split(':')[:-1])]
            if 'WELSPECS' in data:
                groupname = data['WELSPECS']['GROUP'][0]
                try:
                    return self[groupname]
                except KeyError:
                    return WellSegment(parent=self.root, name=groupname, ntype='group', field=self.field)
            return self.root

        for name in sorted(data):
            wdata = data[name]
            name = name.strip(' \t\'"')
            try:
                node = self[name]
            except KeyError:
                parent = _get_parent(name, wdata)
                node = self._nodeclass(parent=parent, name=name, ntype='well', field=self.field)

            if 'WELSPECS' in wdata:
                parent = _get_parent(name, wdata)
                node.parent = parent

            for k, v in wdata.items():
                if mode == 'w':
                    setattr(node, k, v)
                elif mode == 'a':
                    if k in node.attributes:
                        att = getattr(node, k)
                        if isinstance(att, list):
                            setattr(node, k, att+v)
                        else:
                            setattr(node, k, att.append(v, **kwargs))
                    else:
                        setattr(node, k, v)
                else:
                    raise ValueError("Unknown mode {}. Expected 'w' (write) or 'a' (append)".format(mode))
                att = getattr(node, k)
                if isinstance(att, pd.DataFrame) and 'DATE' in att.columns:
                    att = att.sort_values(by='DATE').reset_index(drop=True)
                    setattr(node, k, att)
        return self

    def drop_incomplete(self, logger=None, required=None):
        """Drop nodes with missing 'WELLTRACK' and 'PERF'.

        Parameters
        ----------
        logger : logger, optional
            Logger for messages.
        required: list
            Required attributes for wells. Default ['WELLTRACK', 'PERF'].

        Returns
        -------
        wells : Wells
            Wells without incomplete nodes.
        """
        if required is None:
            required = ['WELLTRACK', 'PERF']
        for node in self:
            if not (('COMPDAT' in node.attributes) or ('COMPDATL' in node.attributes)):
                if not set(required).issubset(node.attributes):
                    self.drop(node.name)
                    if logger is not None:
                        logger.info('Node %s is incomplete and is removed.' % node.name)
        return self

    def drop_outside(self, keep_ancestors=False, logger=None):
        """Drop nodes with missing 'BLOCKS' (outside of the grid).

        Parameters
        ----------
        keep_ancestors : bool
            Keep all ancestors segments for a segment with nonempty 'BLOCKS'. Setting True might result
            in segments with empty 'BLOCKS', e.g. if a parent has no 'BLOCKS' but a child has nonempty
            'BLOCKS'. If False, welltracks may be discontinued. Default False.
        logger : logger, optional
            Logger for messages.

        Returns
        -------
        wells : Wells
            Wells without outside nodes.
        """
        def logger_print(node):
            if logger is not None:
                logger.info(f'Segment {node.name} is outside the grid and is removed.')

        for node in PostOrderIter(self.root):
            if node.is_root and node.name == 'FIELD':
                continue
            if (node.ntype == 'well') and (len(node.blocks) == 0):
                if node.is_leaf:
                    node.parent = None
                    logger_print(node)
                else:
                    if keep_ancestors:
                        continue
                    p = node.parent
                    p.children = list(p.children) + list(node.children)
                    node.parent = None
                    logger_print(node)
        return self

    @apply_to_each_segment
    def add_welltrack(self, segment):
        """Reconstruct welltrack from COMPDAT table.

        To connect the end point of the current segment with the start point of the next segment
        we find a set of segments with nearest start point and take a segment with the lowest depth.
        Works fine for simple trajectories only.
        """
        if ('WELLTRACK' in segment) or ('COMPDAT' not in segment and 'COMPDATL' not in segment):
            return self
        grid = self.field.grid
        if 'COMPDAT' in segment:
            df = segment.COMPDAT[['I', 'J', 'K1', 'K2']].drop_duplicates().sort_values(['K1', 'K2'])
        else:
            if (segment.COMPDATL['LGR']!='GLOBAL').any():
                raise ValueError('LGRs other than `Global` are not supported.')
            df = segment.COMPDATL[['I', 'J', 'K1', 'K2']].drop_duplicates().sort_values(['K1', 'K2'])

        i0, j0 = segment.WELSPECS[['I', 'J']].values[0]
        i0 = i0 if i0 is not None else 0
        j0 = j0 if j0 is not None else 0
        root = np.array([i0, j0, 0])
        track = []
        for _ in range(len(df)):
            dist = np.linalg.norm(df[['I', 'J', 'K1']] - root, axis=1)
            row = df.iloc[[dist.argmin()]]
            xyz = grid.get_xyz([int(row.iloc[0]['I'])-1,
                                int(row.iloc[0]['J'])-1,
                                int(row.iloc[0]['K1'])-1])
            track.append(xyz[:, :4].mean(axis=-2).ravel())
            xyz = grid.get_xyz([int(row.iloc[0]['I'])-1,
                                int(row.iloc[0]['J'])-1,
                                int(row.iloc[0]['K2'])-1])
            track.append(xyz[:, 4:].mean(axis=-2).ravel())
            root = row[['I', 'J', 'K2']].values.astype(float).ravel()
            df = df.drop(row.index)
        track = pd.DataFrame(track).drop_duplicates().values
        segment.WELLTRACK = np.concatenate([track, np.full((len(track), 1), np.nan)], axis=1)
        return self

    @apply_to_each_segment
    def get_blocks(self, segment, logger=None):
        """Calculate grid blocks for the tree of wells.

        Parameters
        ----------
        kwargs : misc
            Any additional named arguments to append.

        Returns
        -------
        comp : Wells
            Wells component with calculated grid blocks and well in block projections.
        """
        grid = self.field.grid

        if 'COMPDAT' in segment.attributes or 'COMPDATL' in segment.attributes:
            if 'COMPDAT' in segment.attributes:
                compdat = segment.compdat
            elif (segment.compdatl['LGR']=='GLOBAL').all():
                compdat = segment.compdatl
            else:
                logger.warning('Well {}: can not get blocks from COMPDATL data.'.format(segment.name))
                return self

            segment.blocks = get_wellblocks_compdat(compdat)
            if isinstance(self.field.grid, OrthogonalGrid):
                h_well = np.stack([(0, 0, self.field.grid.dz[i[0], i[1], i[2]])
                                   for i in segment.blocks])
            else:
                h_well = np.full(segment.blocks.shape, np.NaN)
            segment.blocks_info = pd.DataFrame(h_well, columns=['Hx', 'Hy', 'Hz'])

        else:
            blocks, points, mds = get_wellblocks_vtk(segment.welltrack, grid)

            segment.blocks = blocks
            h_well = abs(points[:, 1] - points[:, 0])
            segment.blocks_info = pd.DataFrame(h_well, columns=['Hx', 'Hy', 'Hz'])
            segment.blocks_info['MDU'] = mds[:, 0]
            segment.blocks_info['MDL'] = mds[:, 1]
            segment.blocks_info['Enter_point'] = list(points[:, 0])
            segment.blocks_info['Leave_point'] = list(points[:, 1])

        segment.blocks_info = segment.blocks_info.assign(
            PERF_RATIO=None if len(segment.blocks_info) == 0 else 0,
            RAD=None if len(segment.blocks_info) == 0 else DEFAULTS['RAD'],
            SKIN=None if len(segment.blocks_info) == 0 else DEFAULTS['SKIN'],
            MULT=None if len(segment.blocks_info) == 0 else DEFAULTS['MULT'],
        )
        return self

    def show_wells(self, figsize=None, c='r', **kwargs):
        """Return 3D visualization of wells.

        Parameters
        ----------
        figsize : tuple
            Output figsize.
        c : str
            Line color, default red.
        kwargs : misc
            Any additional kwargs for plot.
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        for segment in self:
            arr = segment.welltrack
            ax.plot(arr[:, 0], arr[:, 1], arr[:, 2], c=c, **kwargs)
            ax.text(*arr[0, :3], s=segment.name)

        ax.invert_zaxis()
        ax.view_init(azim=60, elev=30)

    def show_rates(self, timesteps=None, wellnames=None, wells2=None, labels=None, figsize=(16, 6)):
        """Plot total or cumulative liquid and gas rates for a chosen node including branches.

        Parameters
        ----------
        timesteps : list of Timestamps
            Dates at which rates were calculated.
        wellnames : array-like
            List of wells to show.
        figsize : tuple
            Figsize for two axes plots.
        wells2 : Wells
            Target model to compare with.
        """
        timesteps = self.result_dates if timesteps is None else timesteps
        wellnames = [node.name for node in PreOrderIter(self.root)]
        return show_rates(self, timesteps=timesteps, wellnames=wellnames, wells2=wells2,
                          labels=labels, figsize=figsize)

    def show_blocks_dynamics(self, timesteps=None, wellnames=None, figsize=(16, 6)):
        """Plot liquid or gas rates and pvt props for a chosen block of
        a chosen well segment on two separate axes.

        Parameters
        ----------
        timesteps : list of Timestamps
            Dates at which rates were calculated.
        wellnames : array-like
            List of wells to plot.
        figsize : tuple
            Figsize for two axes plots.
        """
        timesteps = self.result_dates if timesteps is None else timesteps
        wellnames = self.names if wellnames is None else wellnames
        return show_blocks_dynamics(self, timesteps=timesteps, wellnames=wellnames, figsize=figsize)

    @apply_to_each_segment
    def fill_na(self, segment, attr):
        """
        Fill nan values in wells segment attribute.

        Parameters
        ----------
        attr: str
            Attribute name.

        Returns
        -------
        comp : Wells
            Wells with fixed attribute.
        """
        if attr in segment.attributes:
            data = getattr(segment, attr)
            welspecs = segment.welspecs
            if set(('I', 'J')).issubset(set(data.columns)):
                data['I'] = data['I'].replace(INT_NAN, welspecs['I'].values[0])
                data['J'] = data['J'].replace(INT_NAN, welspecs['J'].values[0])
        return self

    def _read_buffer(self, buffer, attr, **kwargs):
        """Load well data from an ASCII file.

        Parameters
        ----------
        buffer : StringIteratorIO
            Buffer to get string from.
        attr : str
            Target keyword.

        Returns
        -------
        comp : Wells
            Wells component with loaded well data.
        """
        if attr == 'WELSPECS':
            return load_welspecs(self, buffer, **kwargs)
        if attr == 'WELSPECL':
            return load_welspecl(self, buffer, **kwargs)
        if attr == 'COMPDAT':
            return load_compdat(self, buffer, **kwargs)
        if attr == 'COMPDATL':
            return load_compdatl(self, buffer, **kwargs)
        if attr == 'COMPDATMD':
            return load_comdatmd(self, buffer, **kwargs)
        if attr == 'WCONPROD':
            return load_wconprod(self, buffer, **kwargs)
        if attr == 'WCONINJE':
            return load_wconinje(self, buffer, **kwargs)
        if attr == 'WEFAC':
            return load_wefac(self, buffer, **kwargs)
        if attr == 'WFRAC':
            return load_wfrac(self, buffer, **kwargs)
        if attr == 'WFRACP':
            return load_wfracp(self, buffer, **kwargs)
        if attr in ["TFIL", "WELLTRACK"]:
            return load_welltracks(self, buffer, **kwargs)
        if attr in ["EFIL", "EFILE", "ETAB"]:
            return load_events(self, buffer, **kwargs)
        if attr in ["HFIL", "HFILE", "HTAB"]:
            return load_history(self, buffer, **kwargs)
        if attr in ["GROU", "GROUP"]:
            return load_group(self, buffer, **kwargs)
        if attr == "GRUPTREE":
            return load_grouptree(self, buffer, **kwargs)
        if attr == "NETWORK":
            return load_network(self, buffer, **kwargs)
        if attr == "NETBALAN":
            return load_netbalan(self, buffer, **kwargs)
        if attr == "BRANPROP":
            return load_branprop(self, buffer, **kwargs)
        if attr == "NODEPROP":
            return load_nodeprop(self, buffer, **kwargs)
        if attr in ["VFPPROD", "VFPINJ"]:
            return load_vfp(self, buffer, attr, **kwargs)
        if attr == "VFPTABL":
            return load_vfptabl(self, buffer, **kwargs)
        if attr in ["NGASREM", "NWATREM"]:
            return load_nliqrem(self, buffer, clause=attr, **kwargs)
        if attr == "WELTARG":
            return load_weltarg(self, buffer, **kwargs)
        if attr == "WELOPEN":
            return load_welopen(self, buffer, **kwargs)
        raise ValueError("Keyword {} is not supported in Wells.".format(attr))

    def _load_rsm(self, *args, **kwargs):
        """Load RSM well data from file."""
        return load_rsm(self, *args, **kwargs)

    def _load_ecl_binary(self, *args, **kwargs):
        """Load results from UNSMRY file."""
        return load_ecl_binary(self, *args, **kwargs)

    def _dump_ascii(self, path, attr, mode='w', **kwargs):
        """Save data into text file.

        Parameters
        ----------
        path : str
            Path to output file.
        attr : str
            Attribute to dump into file.
        mode : str
            Mode to open file.
            'w': write, a new file is created (an existing file with
            the same name would be deleted).
            'a': append, an existing file is opened for reading and writing,
            and if the file does not exist it is created.
            Default to 'w'.

        Returns
        -------
        comp : Wells
            Wells unchanged.
        """
        with open(path, mode) as f:
            if attr.upper() == 'WELLTRACK':
                for node in self:
                    if 'WELLTRACK' in node and 'COMPDAT' not in node and 'COMPDATL' not in node:
                        f.write('WELLTRACK\t{}\n'.format(node.name))
                        for line in node.welltrack:
                            f.write(' '.join(line.astype(str)) + '\n')
            elif attr.upper() == 'PERF':
                write_perf(f, self, DEFAULTS)
            elif attr.upper() == 'GROUP':
                for node in PreOrderIter(self.root):
                    if node.is_root:
                        continue
                    if node.ntype == 'group' and not node.is_leaf and not node.children[0].ntype == 'group':
                        f.write(' '.join(['GROUP', node.name] +
                                         [child.name for child in node.children]) + '\n')
                f.write('/\n')
            elif attr.upper() == 'GRUPTREE':
                # Extended network keywords must be placed together.
                # Order follows your example deck snippet.
                write_netbalan(f, self)
                f.write('GRUPTREE\n')
                for node in PreOrderIter(self.root):
                    if node.is_root:
                        continue
                    if node.ntype == 'group' and node.parent.ntype == 'group':
                        p_name = '1*' if node.parent.is_root else node.parent.name
                        f.write(' '.join([node.name, p_name, '/\n']))
                f.write('/\n')
                write_branprop(f, self)
                write_nodeprop(f, self)
                write_network(f, self)
            elif attr.upper() == 'EVENTS':
                write_events(f, self, VALUE_CONTROL)
            elif attr.upper() == 'SCHEDULE':
                write_schedule(f, self, **kwargs)
            elif attr.upper() == 'WELSPECS':
                write_welspecs(f, self)
            else:
                raise NotImplementedError("Dump for {} is not implemented.".format(attr.upper()))

class Network(Wells):
    """Network component.

    Contains wells, groups and surface network nodes in one structure.

    Parameters
    ----------
    node : WellSegment, optional
        Root node for well's tree.
    """
    def __init__(self, node=None, **kwargs):
        super().__init__(node=node, **kwargs)
        self._sink = None
        self._net_nodes = None
        self._n_nodes = None
        self.ipr_dict = None
        self._node2index = None
        self._wellnames = sorted([w.name for w in self])
        self._netcols = []
        self._rate_bounds = None
        self._chok_bounds = None
        self._default_net_ipr = {}
        self._default_ipr_dict = {}
        self._autochokes = []
        self._network_solution = None
        self._network_optimizer_config = {'method': 'SLSQP'}
        self._kwargs = kwargs
        self._shutoff_wells = []
        self.gor_dict = {}
        self.wct_dict = {}

    def _init_network(self):
        """Initialize network by setting all the necessary data and structures from kwargs for fast iteration.
        """
        self._set_sink(self._kwargs.get('sink', None))
        self._default_net_ipr = {
            'func': self._default_ipr_function,
            'inv_func': self._default_ipr_inv_function,
            'Pr': 181,
            'J': 3
        }
        self._wellnames = sorted(self._select_nodes(lambda x: x.ntype == "well", root=self._sink, attr="name"))
        self._default_ipr_dict = {w:dict(**self._default_net_ipr) for w in self._wellnames}
        self._set_ipr(self._kwargs.get('ipr_config', self._default_ipr_dict))
        self._netcols = ['LIQUID_RATE', 'WATER_CUT', 'GAS_OIL_RATIO', 'WATER_RATE',
                         'GAS_RATE', 'PRESSURE', 'CHOKE', 'NWATREM', 'NGASREM']
        self._reset_net_rates()
        self._set_rates()
        self._set_chokes(self._kwargs.get("choke", None))
        self._set_liqrem()
        self._set_autochokes(self._kwargs.get("autochokes", None))
        self._set_network_optimizer_config()
        self._rate_bounds = [(0, 500) for _ in self._wellnames]
        self._set_rate_constraints(self._kwargs.get('constraints', None))

    def _select_nodes(self,
                      criterion=lambda n: n,
                      attr=False,
                      order='level',
                      root=None,
                      mapping=lambda x: x):
        """Return all nodes which satisfy the criterion."""
        iterators = {'pre': PreOrderIter, 'post': PostOrderIter, 'level': LevelOrderIter}
        if root is None:
            root = self.root
        nodes = list(iterators[order](root, filter_=criterion))
        if not attr:
            return [mapping(n) for n in nodes]
        return [mapping(getattr(n, attr)) for n in nodes]

    def _default_ipr_function(self, q, **kwargs):
        """Linear IPR function."""
        return kwargs['Pr'] - q / kwargs['J']

    def _default_ipr_inv_function(self, P, **kwargs):
        """Linear IPR function."""
        return kwargs['J'] * (kwargs['Pr'] - P)
    #numpy
    @apply_to_each_node(node_types=('well', 'group'), root='_sink')
    def _reset_net_rates(self, node):
        setattr(node, 'net_rates', {col: 0 for col in self._netcols})

    def _set_liqrem(self):
        all_nodes = self._select_nodes(root=self._sink)

        for node in all_nodes:
            if 'NWATREM' in node.attributes:
                self[node.name].net_rates['NWATREM'] = node.nwatrem["MAXRAT"].item()
            else:
                self[node.name].net_rates['NWATREM'] = 0

            if 'NGASREM' in node.attributes:
                self[node.name].net_rates['NGASREM'] = node.ngasrem["MAXRAT"].item()
            else:
                self[node.name].net_rates['NGASREM'] = 0

    def _set_rates(self, rate_dict=None):
        """Fill np.array with rates from dict

        Parameters
        ----------
        rate_dict : dict, optional
            dict of form {"$wellname$":{"$rateparameter1$": value1, ...}, ...}, by default None

        Raises
        ------
        ValueErrorpartial
            When `rateparametern` is not present among supported
        """
        if rate_dict is None:
            return
        for wname, rates in rate_dict.items():
            if wname not in self:
                raise ValueError("No node {} detected".format(wname))
            for key, value in rates.items():
                if key.upper() not in self._netcols:
                    raise ValueError("Rate parameter {} is not supported.".format(key.upper()))
                self[wname].net_rates[key] = value

        if 'PRESS' in self._sink.attributes:
            self._sink.net_rates['PRESSURE'] = self._sink.press

    def _set_sink(self, sink=None):
        """Set or find sink node

        Parameters
        ----------
        sink : str or Segment
            Node or node name, optional, by default None
        rate_limit : int, optional
            rate limit on sink in m3/d, by default 50000

        Raises
        ------
        ValueError
            When name or node is not found among nodes.
        """
        if sink is None:
            self._sink = self._get_sink_from_data()
        elif isinstance(sink, str):
            self._sink = self[sink]
        elif sink in self.root.descendants:
            self._sink = sink
        else:
            raise ValueError("`sink` is not found among nodes.")

    def _set_shutoff_wells(self, names):
        self._shutoff_wells = [name for name in names if name in self.wellnames]

    def _get_sink_from_data(self):
        """Get sink = the first node from the top with `PRESSURE` attribute."""
        for node in LevelOrderIter(self.root):
            if 'PRESS' in node.attributes:
                return node
        raise ValueError("Sink wasn't declared in wells.")

    @property
    def _fixed_pressure_nodes(self):
        """Get sink = the first node from the top with `PRESSURE` attribute."""
        lst = []
        for node in LevelOrderIter(self._sink):
            if 'PRESS' in node.attributes and node.press not in [0, "1*"]:
                lst.append(node)
        return lst

    def _set_ipr(self, ipr_dict=None):
        """Set IPR function and arguments for every well."""
        if not ipr_dict:
            return

        def model_func(x, model, **kwargs):
            _ = kwargs
            return model(x)

        self.ipr_dict = ipr_dict
        for w in self.ipr_dict:
            ipr_params = self.ipr_dict.get(w, self._default_net_ipr)
            if "data" in ipr_params:
                data = ipr_params.pop("data")
                model = interp1d(data[:, 0], data[:, 1], **ipr_params["interp_dict"])
                inv_model = interp1d(data[:, 1], data[:, 0], **ipr_params["interp_dict"])
                ipr_params['func'] = partial(model_func, model=model)
                ipr_params['inv_func'] = partial(model_func, model=inv_model)

            setattr(self[w], 'ipr', ipr_params.pop('func'))
            if 'inv_func' in ipr_params:
                setattr(self[w], 'ipr_inv', ipr_params.pop('inv_func'))
            setattr(self[w], 'ipr_params', ipr_params)

    def _set_wct(self, wct_dict=None):
        """Set wct function and arguments for every well."""
        def model_func(x, model, **kwargs):
            _ = kwargs
            out = model(x)
            out = np.clip(out, 0, 1)
            return out

        if wct_dict in [None, {}, ()]:
            return
        self.wct_dict = wct_dict
        for w in self.wct_dict:
            wct_params = self.wct_dict.get(w, {})
            if "data" in wct_params:
                data = wct_params.pop("data")
                model = interp1d(data[:, 1], data[:, 0], **wct_params["interp_dict"])
                wct_params['func'] = partial(model_func, model=model)
            setattr(self[w], 'wct', wct_params.pop('func'))
            setattr(self[w], 'wct_params', wct_params)

    def _set_gor(self, gor_dict=None):
        """Set gor function and arguments for every well."""
        def model_func(x, model, **kwargs):
            _ = kwargs
            return model(x)

        if gor_dict in [None, {}, ()]:
            return
        self.gor_dict = gor_dict
        for w in self.gor_dict:
            gor_params = self.gor_dict.get(w, {})
            if "data" in gor_params:
                data = gor_params.pop("data")
                model = interp1d(data[:, 1], data[:, 0], **gor_params["interp_dict"])
                gor_params['func'] = partial(model_func, model=model)
            setattr(self[w], 'gor', gor_params.pop('func'))
            setattr(self[w], 'gor_params', gor_params)

    def _set_chokes(self, choke_dict=None):
        """Initialize chokes.
        Parameters
        ----------
        choke_dict : dict, optional
            dictionary of form `{"chokename": press_diff, ...}`, by default None
        autochokes : list, optional
            list of autochokes' names, by default None
        """
        if choke_dict in [None, 0, False]:
            return
        if isinstance(choke_dict, dict):
            for node_name, choke_press in choke_dict.items():
                if node_name not in self:
                    raise ValueError("No node {} detected".format(node_name))
                self[node_name].net_rates['CHOKE'] = choke_press

    def _set_autochokes(self, autochokes=None):
        """Initiate autochokes."""
        if autochokes is None:
            is_autochoke = lambda x: 'CHOKE' in x.attributes and x.choke #pylint: disable=unnecessary-lambda-assignment
            self._autochokes = self._select_nodes(is_autochoke, 'name')
        else:
            self._autochokes = autochokes
        self._chok_bounds = [(0, 100) for a in self._autochokes]
    #optimizer
    def _set_rate_constraints(self, dct=None):
        """Set constraints for the optimizer."""
        if dct is None:
            return
        def rate_limit_fun(x, rate_limit, indices):
            return rate_limit[0] - np.sum(x[indices])

        def rate_limit_jac(x, rate_limit, indices):
            _ = rate_limit
            out = np.zeros(len(x))
            out[indices] = -1
            return out

        def rate_constraint_fun(x, rate_bounds, indices):
            return (np.sum(x[indices]) - rate_bounds[0]) * (rate_bounds[1] - np.sum(x[indices]))

        def rate_constraint_jac(x, rate_bounds, indices):
            out = np.zeros(len(x))
            out[indices] = -1 * (2 * np.sum(x[indices]) - sum(rate_bounds))
            return out

        constraints = []
        for node_name, bounds in dct.items():
            try:
                node = self[node_name]
            except Exception as e:
                raise ValueError("No such node is found.") from e

            if node_name in self.wellnames: #well
                assert len(bounds) in (1, 2), "Bounds should be 1-or-2-tuple."
                self._rate_bounds[self.wellnames.index(node_name)] = bounds
            elif node in self.groups:
                winds = [self.wellnames.index(w.name) for w in node.descendants if w.ntype == 'well']
                if len(bounds) == 1:
                    constraints.append(
                        {"type": "eq",
                         "fun": rate_limit_fun,
                         "jac": rate_limit_jac,
                         "args": (bounds, winds)}
                    )
                elif len(bounds) == 2:
                    constraints.append(
                        {"type": "ineq",
                         "fun": rate_constraint_fun,
                         "jac": rate_constraint_jac,
                         "args": (bounds, winds)}
                    )
                else:
                    raise ValueError("bounds should be 1-or-2-tuple: {}-tuple is given".format(len(bounds)))
        self._network_optimizer_config['constraints'] = constraints

    def _set_network_optimizer_config(self, **config):
        """Set network optimizer config"""
        self._network_optimizer_config = {}
        if "NETBALAN" in self.root.attributes:
            maxiter = int(self.root.netbalan["MAXITER"].item())
            self._network_optimizer_config.update({"options": {"maxiter": maxiter}})

        self._network_optimizer_config.update(**config)
    #numpy
    @property
    def network_table(self):
        """Dataframe formed from network."""
        rates = self._select_nodes(root=self._sink, attr='net_rates')
        names = self._select_nodes(root=self._sink, attr='name')
        return pd.DataFrame(rates, index=names)

    @property
    def wellnames(self):
        """well names list."""
        self._wellnames = sorted(self._select_nodes(lambda x: x.ntype == "well", root=self._sink, attr="name"))
        return self._wellnames

    @wellnames.setter
    def wellnames(self, wnames):
        wellnames = []
        for wname in wnames:
            if wname in self:
                wellnames.append(wname)
        self._wellnames = wellnames

    @property
    def groups(self):
        """Return all groups."""
        return self._select_nodes(lambda n: n.ntype == 'group')

    @apply_to_each_node(node_types=('group',), reverse=True, root='_sink', order="LevelOrderIter")
    def calculate_network_rates(self, node):
        """Impure method that calculates rates up the network."""

        ndf = node.net_rates
        sum_liquid = 0
        sum_water = 0
        sum_gas = 0
        for child in node.children:
            # print(child.name)
            cdf = child.net_rates
            if child.name in self._shutoff_wells:
                continue
            cdf['WATER_RATE'] = cdf['LIQUID_RATE'] * cdf['WATER_CUT']
            cdf['GAS_RATE'] = cdf['LIQUID_RATE'] * (1 - cdf['WATER_CUT']) * cdf['GAS_OIL_RATIO']
            sum_liquid += cdf['LIQUID_RATE'] - cdf["NWATREM"] * cdf['WATER_RATE']
            sum_water += cdf['WATER_RATE'] * (1 - cdf["NWATREM"])
            sum_gas += cdf['GAS_RATE'] * (1 - cdf["NGASREM"])
        if sum_liquid < np.finfo(float).eps:
            ndf['GAS_OIL_RATIO'] = 0
            ndf['LIQUID_RATE'] = 0
            ndf['WATER_RATE'] = 0
            ndf['WATER_CUT'] = 0
            ndf['GAS_RATE'] = 0
            return
        ndf['LIQUID_RATE'] = sum_liquid
        ndf['WATER_RATE'] = sum_water
        ndf['GAS_RATE'] = sum_gas
        ndf['WATER_CUT'] = ndf['WATER_RATE'] / ndf['LIQUID_RATE']
        ndf['GAS_OIL_RATIO'] = 0
        sum_oil_gas = ndf['LIQUID_RATE'] - ndf['WATER_RATE']
        if sum_oil_gas > np.finfo(float).eps:
            ndf['GAS_OIL_RATIO'] = ndf['GAS_RATE'] / sum_oil_gas

    @apply_to_each_node(node_types=('group',), reverse=True, root='_sink', order="LevelOrderIter")
    def _determine_detached(self, node):
        """Impure method that calculates rates up the network."""
        setattr(node, "detach", True)
        for child in node.children:
            node.detach &= child.detach

    @apply_to_each_node(node_types=('group',), root='_sink', order="LevelOrderIter")
    def calculate_network_pressures(self, node):
        """Impure method that calculates pressures down the network."""
        if node in self._fixed_pressure_nodes:
            node.net_rates['PRESSURE'] = node.press

        for child in node.children:
            vfp_arg = [child.net_rates['LIQUID_RATE'],
                       node.net_rates['PRESSURE'] + node.net_rates['CHOKE'],
                       child.net_rates['WATER_CUT'],
                       child.net_rates['GAS_OIL_RATIO']]

            if "ALQ_NODE" in child.attributes and child.alq_node:
                vfp_arg.append(child.alq_node)
            if 'ALQ' in child.attributes and child.alq and child.vfp.meta["ALQ"] not in (" ", "", "False"):
                if len(vfp_arg) == 5:
                    vfp_arg[-1] = child.alq
                else:
                    vfp_arg.append(child.alq)

            vfp_bhp = child.vfp(vfp_arg).item()
            child.net_rates['PRESSURE'] = vfp_bhp

    def update_wct_gor(self):
        """Update water cut and gor for wells if they have appropriate methods."""
        for w in self.wellnames:
            if hasattr(self[w], "wct"):
                self[w].net_rates['WATER_CUT'] = self[w].wct(self[w].net_rates['PRESSURE'], **self[w].wct_params)

            if hasattr(self[w], "gor"):
                self[w].net_rates['GAS_OIL_RATIO'] = self[w].gor(self[w].net_rates['PRESSURE'], **self[w].gor_params)

    def calculate_ipr(self):
        """Update well liquid rate if well has appropriate method."""
        for w in self.wellnames:
            if hasattr(self[w], "ipr") and hasattr(self[w], "ipr_params"):
                self[w].net_rates['LIQUID_RATE'] = self[w].ipr(self[w].net_rates['PRESSURE'], **self[w].ipr_params)

    def calculate_network_error(self):
        """Calculate pressure error."""
        out = [self[n].ipr(self[n].net_rates['LIQUID_RATE'], **self[n].ipr_params)-
               self[n].net_rates['PRESSURE'] for n in self.wellnames]
        out = np.abs(out) ** 2
        out = np.sum(out)
        return out

    def iteration_error(self, guess=None):
    # def iteration_error(self, guess, opt_value):
        """Make a pass up and down network and return error.
        Parameters
        ----------
        guess : (l+c,) array_like
            concatenation of liquid rates and autochoke pressures.
        Returns
        ----------
        err : float
            Error at this iteration.
        """
        if guess is not None:
            nwells = len(self.wellnames)
            for i, well_name in enumerate(self.wellnames):
                self[well_name].net_rates['LIQUID_RATE'] = guess[i]
            if self._autochokes:
                for j, autochoke in enumerate(self._autochokes):
                    self[autochoke].net_rates['CHOKE'] = guess[nwells+j]

        self.calculate_network_rates()
        self.calculate_network_pressures()
        return self.calculate_network_error()

    def _generate_guess(self, x_bounds):
        lower_bound, upper_bound = zip(*x_bounds)
        return np.random.uniform(lower_bound, upper_bound)

    def total_network_analysis(self, guess_0=None, chokes_0=None, **kwargs):
        """
        Balances the network and returns solution.
        Parameters
        ----------
        guess_0 : (l,) array_like
            array of well liquid rate or bhp initial guesses, `l = len(self.wellnames)`
        chokes_0 : (c,) array_like, optional
            array of autochoke pressure difference initial guesses, `c = len(self._autochokes)`
            by default []
        Returns
        -------
        solution : (l+c,) np.ndarray
            array of solution
        """
        self._network_optimizer_config['bounds'] = self._rate_bounds + self._chok_bounds
        self._network_optimizer_config.update(**kwargs)
        if guess_0 is None:
            guess_0 = self._generate_guess(self._rate_bounds)
        if chokes_0 is None:
            chokes_0 = self._generate_guess(self._chok_bounds) if self._autochokes else []
        nwells = len(list(self))
        guess = np.concatenate((guess_0, chokes_0))

        def min_callback(x):
            _ = x

            self.update_wct_gor()

        self._network_solution = minimize(self.iteration_error,
                                          guess,
                                          callback=min_callback,
                                          **self._network_optimizer_config).x

        for i, well_name in enumerate(self.wellnames):
            try:
                w = self[well_name]
            except Exception as e:
                raise ValueError("No node with this name: {}".format(well_name)) from e
            w.net_rates['LIQUID_RATE'] = self._network_solution[i]
        if self._autochokes:
            for i, choke_name in enumerate(self._autochokes):
                try:
                    achoke = self[choke_name]
                except Exception as e:
                    raise ValueError("No node with this name: {}".format(choke_name)) from e
                achoke.net_rates['CHOKE'] = self._network_solution[nwells+i]
        for name in self._shutoff_wells:
            n_r = self[name].net_rates
            n_r['LIQUID_RATE'] = 0
            n_r['WATER_RATE'] = 0
            n_r['GAS_RATE'] = 0
            n_r['GAS_OIL_RATIO'] = 0
            n_r['WATER_CUT'] = 0
        return self._network_solution
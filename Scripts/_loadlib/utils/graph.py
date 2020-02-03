import sys
import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (6,5)
plt.rcParams["axes.edgecolor"] = "0.15"
plt.rcParams["axes.linewidth"] = 1.25
plt.rcParams["figure.facecolor"] = "white"
try:
    from hiveplot import HivePlot
except ImportError:
    print('Missing dependencies in hiveplot module, not loaded')
try:
    import graph_tool.all as gt
except ImportError:
    print('Missing dependencies in graph_tool module, not loaded')

class draw():
    def __init__(self, adj_mat=[], mode='fr', node_color=None, edge_color=[],
                 cmap='gnuplot', output='', figsize=(500,500), vertex_size=5,
                 edge_cmap='Greys', cooling_step=0.99, edge_alpha=1.0,
                 epsilon=1e-2, min_span=False, random_state=None,
                 directed=True, vertex_text='None'):
        self.size = np.shape(adj_mat)[0]
        if isinstance(adj_mat, pd.DataFrame):
            self.node_index = adj_mat.index.values
            self.adj_mat = adj_mat.values
        else:
            self.adj_mat = np.array(adj_mat)
            self.node_index = np.arange(self.size)
        self.mode = mode
        self.output = output
        self.cmap = cmap
        self.figsize=figsize
        self.vertex_size = vertex_size
        self.vertex_text = vertex_text
        self.cooling_step = cooling_step
        self.epsilon = epsilon
        self.min_span = min_span
        self.random_state = random_state
        if node_color is None:
            self.node_color = np.zeros(self.size)
        else:
            self.node_color = self.convert_str_to_color(node_color)
        self.edge_cmap = edge_cmap
        self.edgelist = np.transpose(self.adj_mat.nonzero())
        self.directed = directed
        if type(edge_color) not in [int, float]:
            self._edge_colorcode = 0.2
            if len(edge_color) == 0:
                self.edge_color = []
            else:
                if np.ndim(edge_color) == 1:
                    self.edge_color = self.convert_str_to_color(edge_color)
                else:
                    tmp_edge_color = np.array(edge_color)
                    np.fill_diagonal(tmp_edge_color, 0)
                    if np.min(tmp_edge_color.flatten()) < 0:
                        tmp_edge_color[tmp_edge_color < 0] /= np.abs(tmp_edge_color.min())
                    tmp_edge_color[tmp_edge_color > 0] /= np.abs(tmp_edge_color.max())
                    tmp_edge_color -= np.min(tmp_edge_color)
                    tmp_edge_color /= np.max(tmp_edge_color)
                    self._tmp_edge_color = tmp_edge_color
                    self.edge_color = np.zeros(np.shape(self.edgelist)[0])
        else:
            self._edge_colorcode = edge_color
            self.edge_color = []
        self.edge_alpha = edge_alpha

    def run(self, **args):
        self.zero_out_diag()
        self.make_graph()
        if self.min_span == True:
            self.make_min_span_tree()
        self.generate_pos(**args)
        if self.directed == False:
            self.g.set_directed(False)

    def plot(self, weigh_edge=False, edge_width=0.8, **args):
        self.edge_width = edge_width
        if isinstance(edge_width, str):
            self.weigh_edge_betweeness(scaling_factor=weigh_edge)
            width_prop = self.be
        else:
            width_prop = self.g.new_edge_property('double')
            if np.ndim(edge_width) < 2:
                repeat_width = True
            else:
                repeat_width = False
            for eid, i in enumerate(self.g.edges()):
                source, target = int(i.source().__str__()), int(i
                                                      .target().__str__())
                if repeat_width:
                    width_prop[i] = edge_width
                else:
                    width = float(edge_width[source, target])
                    width_prop[i] = width
        self.g.edge_properties['width'] = width_prop
        if self.vertex_text.lower() == 'name':
            vertex_text=self.g.vertex_properties['name']
        elif self.vertex_text.lower() == 'none':
            vertex_text = None
        else:
            vertex_text=self.g.vertex_index
        if self.mode in ['sfdp', 'fr', 'arf']:
            if self.output != '':
                gt.graph_draw(self.g, pos=self.g.vertex_properties['pos'],
                              vertex_fill_color=
                              self.g.vertex_properties['fill_color'],
                              output_size=self.figsize*2,
                              vertex_text=vertex_text,
                              edge_color = self.g.edge_properties['color'],
                              edge_pen_width = self.g.edge_properties['width'],
                              output=self.output, dpi=500)
                plt.close()
            else:
                 gt.graph_draw(self.g, pos=self.g.vertex_properties['pos'],
                               vertex_fill_color=
                               self.g.vertex_properties['fill_color'],
                               vertex_text=vertex_text,
                               output_size=self.figsize,
                               edge_pen_width = self.g.edge_properties['width'],
                               edge_color = self.g.edge_properties['color'])
        elif self.mode == 'radial':
            if self.output != '':
                gt.graph_draw(self.g, pos=self.g.vertex_properties['pos'],
                              edge_control_points=self.g.edge_properties['cts'],
                              vertex_text=vertex_text,
                              vertex_fill_color=self.g.vertex_properties[
                                  'fill_color'],edge_color=self.g.edge_properties[
                                      'color'], output_size=self.figsize*2,
                              random_state=self.random_state,
                              edge_pen_width = self.g.edge_properties['width'],
                              output=self.output, dpi=500, **args)
                plt.close()
            else:
                gt.graph_draw(self.g, pos=self.g.vertex_properties['pos'],
                              edge_control_points=self.g.edge_properties['cts'],
                              vertex_text=vertex_text,
                              vertex_fill_color=self.g.vertex_properties[
                              'fill_color'],edge_color=self.g.edge_properties[
                                      'color'], output_size=self.figsize,
                              edge_pen_width = self.g.edge_properties['width'],
                              random_state=self.random_state, **args)
        else:
            print('provided "mode" is not fr, radial, or sfdp')

    def zero_out_diag(self):
        self.adj_mat[range(self.size), range(self.size)] = 0

    def make_graph(self):
        assert self.mode in ['fr', 'radial', 'sfdp', 'arf'], 'Provided mode must be either "fr", "radial", or "sfdp"'
        self.g = gt.Graph()
        vprop = self.g.new_vertex_property("string")
        self.g.vertex_properties['name'] = vprop
        self.g.add_vertex(len(self.node_index))
        for i, index in enumerate(self.node_index):
            vprop[self.g.vertex(i)] = index
        self.g.add_edge_list(self.edgelist)
        if self.mode not in ['radial'] or self.node_color.sum() != 0:
            self.add_color_nodes()
        self.add_color_edges()

    def generate_pos(self, **args):
        if self.mode == 'fr':
            self.make_fr_graph(**args)
        elif self.mode == 'radial':
            self.make_radial_graph(**args)
        elif self.mode == 'sfdp':
            self.make_sfdp_graph(**args)
        elif self.mode == 'arf':
            self.make_arf_graph(**args)
        else:
            print('provided "mode" must be either fr, radial, sfdp, or arf')

    def convert_str_to_color(self, v):
        labels = np.array(v).astype(str)
        tick_dictionary = dict([(y,x) for x,y in
                                enumerate(sorted(set(np.unique(labels))))])
        c = np.array([tick_dictionary[x] for x in labels])
        c = c / np.max(c)
        return c

    def add_color_nodes(self):
        color_prop = self.g.new_vertex_property('vector<double>')
        for eid, i in enumerate(self.g.vertices()):
            color_prop[i] = plt.get_cmap(self.cmap)(self.node_color[eid])
        self.g.vertex_properties['fill_color'] = color_prop

    def add_color_edges(self):
        color_prop = self.g.new_edge_property('vector<double>')
        if len(self.edge_color) == 0:
            for eid, i in enumerate(self.g.edges()):
                r,g,b,alpha = plt.get_cmap(self.edge_cmap)(
                    self._edge_colorcode)
                color_prop[i] = (r,g,b,self.edge_alpha)
        else:
            for eid, i in enumerate(self.g.edges()):
                source, target = int(i.source().__str__()), int(i
                                                          .target().__str__())
                color = self._tmp_edge_color[target, source]
                r,g,b,alpha = plt.get_cmap(self.edge_cmap)(color)
                self.edge_color[eid] = color
                color_prop[i] = (r,g,b,self.edge_alpha)
        self.g.edge_properties['color'] = color_prop

    def make_fr_graph(self):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        pos = self.g.new_edge_property("double")
        self.g.vertex_properties['pos'] = gt.fruchterman_reingold_layout(self.g)

    def make_arf_graph(self, **args):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        pos = self.g.new_edge_property("double")
        self.g.vertex_properties['pos'] = gt.arf_layout(self.g, **args)

    def make_radial_graph(self, **args):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        state = gt.minimize_nested_blockmodel_dl(self.g, deg_corr=True)
        t = gt.get_hierarchy_tree(state)[0]
        tpos = gt.radial_tree_layout(t, t.vertex(t.num_vertices() -
                                                            1), weighted=True)
        cts = self.g.new_edge_property("double")
        pos = self.g.new_edge_property("double")
        self.g.edge_properties['cts'] = gt.get_hierarchy_control_points(self.g,
                                                                    t, tpos)
        self.g.vertex_properties['pos'] = self.g.own_property(tpos)
        self.membership = list(state.get_bs()[0])
        if self.node_color.sum() == 0:
            self.node_color = self.convert_str_to_color(self.membership)
            self.add_color_nodes()

    def make_sfdp_graph(self):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        pos = self.g.new_edge_property("double")
        self.g.vertex_properties['pos'] = (gt.sfdp_layout(self.g,
                                              cooling_step=self.cooling_step,
                                                epsilon=self.epsilon))

    def make_min_span_tree(self):
        tree = gt.min_spanning_tree(self.g)
        self.g.set_edge_filter(tree)

    def weigh_edge_betweeness(self, scaling_factor=5):
        self.bv, self.be = gt.betweenness(self.g)
        self.be.a /= self.be.a.max() / scaling_factor

    def save(self, savename, exc=None):
        """Saves all graph attributes to a Pickle file.

        Saves all graph attributes to a Pickle file which can be later loaded
        into an empty graph object.

        Parameters
        ----------
        savename - string
            The name of the pickle file (not including the file extension) to
            write to.

        exc - array-like of strings, optional, default None
            A vector of graph attributes to exclude from the saved file. Use this
            to exclude bulky objects that do not need to be saved.

        """
        keys = self.__dict__.keys()
        pickle_dict = {}
        for i in list(keys):
            if i != 'g':
                pickle_dict[i] = self.__dict__[i]
        self.g.save(savename.rstrip('.p')+'.xml')
        try:
            del pickle_dict['g']
        except:
            0;
        if savename[-2:] != '.p':
            savename = savename + '.p'
        f = open(savename, 'wb')
        pickle.dump(pickle_dict, f)
        f.close()

    def load(self, n):
        """Loads graph attributes from a Pickle file.

        Loads all graph attributes from the specified Pickle file into the graph
        object.

        Parameters
        ----------
        n - string
            The path of the Pickle file.
        """
        n = n.rstrip('.p')+'.p'
        f = open(n, 'rb')
        pick_dict = pickle.load(f)
        for i in range(len(pick_dict)):
            self.__dict__[list(pick_dict.keys())[i]
                          ] = pick_dict[list(pick_dict.keys())[i]]
        f.close()
        self.g = gt.Graph()
        self.g.load(n.replace('.p', '.xml'))


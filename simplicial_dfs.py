import pandas as pd 
import os 

"""we maintain a datastructure of a clique complex with the assumption
      for each vertex there is an edge with it as source or target = no isolated points
one puts in a generating edge list < source | target | data(?) > and the Clique-Complex
is built from it as a convenient data structure of max cliques as well as degree wise
cliques .. in principle one could build inclusions of such things, that's interesting 
enough for business, to see cycles close for example

the edgelist dataframe can be considered directed as well as undirected, it will handle
itself conveniently with a total order on union{source-type, target-type} e.g. string or int
like building cliques on the symmetrised dataframe and using the total order to give ordered
tuples back 
furthermore it will maintain a dataframe of vertex labels which is assumed to have data like
    vertex_labels_df = < index | label | [data] > 
the index is maintained aligned with the total ordering on "label" 
as well as data on edges 
    edgelist_df = < source_index | target_index | [data] >
every other data structure is assumed computed from this here code, not imported from source-frames
but will carry computed data like lagrangian eigenvectors, cycles.. and such"""

class config_handler():
    @staticmethod
    def to_key_value(d:dict, filename:str):
        pd.DataFrame(d.items(), columns=['key','value']).to_csv(filename,index=False)

    @staticmethod
    def from_key_value(filename:str, flipped:bool=False)->dict:
        df = pd.read_csv(filename)
        if flipped:
            return dict(zip(df.value, df.key))
        return dict(zip(df.key, df.value))

    def __init__(self, name:str='pathing_stub.csv'):
        folder, filename = '/'.join(name.split('/')[:-1]), name.split('/')[-1]
        folder = '.' if len(folder)==0 else folder
        if filename in os.listdir(folder):
            self.config = config_handler.from_key_value(name)
        else: 
            self.config = dict()
            config_handler.to_key_value(filename)
        self.filename = name

    def __repr__(self):
        return 'filenamed_config at: \n\t'+self.filename\
             + '\nwith values:\n\t'\
             + '\n\t'.join(str(k)+': '+str(v) for k,v in self.config.items() )

    def is_key_set(self, key:str)->bool:
        self.config = config_handler.from_key_value(self.filename)
        return key in self.config.keys()

    def get_key(self, key:str):
        self.config = config_handler.from_key_value(self.filename)
        assert key in self.config.keys(), "key unknown"
        tmp = self.config[key]
        return tmp

    def set_new_key(self, key:str, value):
        self.config = config_handler.from_key_value(self.filename)
        assert key not in self.config.keys(), "key known, won't set"
        self.config.update({key:value})
        config_handler.to_key_value(self.config, self.filename)

    def set_update_key(self, key:str, value):
        self.config = config_handler.from_key_value(self.filename)
        assert key in self.config.keys(), "key unknown, won't update"
        self.config[key]=value
        config_handler.to_key_value(self.config, self.filename)

    @classmethod
    def read_from_full_filename(cls,filename:str='pathing_stub.csv'):
        return cls(name=filename)

FOLDER_CONFIG_FILENAME = "pathing_stub.csv"
config_stub = config_handler.read_from_full_filename(filename=FOLDER_CONFIG_FILENAME)
data_folder = config_stub.get_key('complex_data_folder')
if data_folder not in os.listdir():
    os.mkdir(data_folder)
del FOLDER_CONFIG_FILENAME    

class complex_explorer_from_graph():    
    def __init__(self, name:str='__pathing_dev'):
        """
        this hooks our explorer up to its file infrastructure according to
        a self.config, everything else is done in loading, computing, etc methods
        """
        def __set_up_pathing__(NAMED_ROOT:str):                        
            config_file = NAMED_ROOT + 'pathing.csv'
            RAW_FOLDER = NAMED_ROOT + config_stub.get_key('raw_folder')
            RAW_FOLDER_PLUS_RAW_INFIX = RAW_FOLDER + config_stub.get_key('raw_file_prefix')
            COMPUTED_FOLDER = NAMED_ROOT + config_stub.get_key('computed_folder_suffix')
            COMPUTED_VERTEX_FOLDER = COMPUTED_FOLDER + config_stub.get_key('computed_vertex_folder_suffix')
            COMPUTED_EDGE_FOLDER = COMPUTED_FOLDER + config_stub.get_key('computed_edge_folder_suffix')
            config =\
                {
                    'name': name,                     
                    'NAMED_ROOT': NAMED_ROOT,
                    'RAW_FOLDER': RAW_FOLDER,
                    'RAW_FOLDER_PLUS_RAW_INFIX': RAW_FOLDER_PLUS_RAW_INFIX,
                    'COMPUTED_FOLDER': COMPUTED_FOLDER,
                    'COMPUTED_VERTEX_FOLDER': COMPUTED_VERTEX_FOLDER,
                    'COMPUTED_EDGE_FOLDER': COMPUTED_EDGE_FOLDER                    
                }
            os.mkdir(config['NAMED_ROOT'])          # create ./sx_path/<name>/ root folder
            config_handler.to_key_value(config, config_file)     # create ./sx_path/<name>/pathing.csv meta information on folder_structures
            os.mkdir(config['RAW_FOLDER'])                 # create ./sx_path/<name>/raw/ folder for raw_0.csv + raw_1.csv
            os.mkdir(config['COMPUTED_FOLDER'])            # create ./sx_path/<name>/computed/ folder for <n>_df-folders to go in (or even (n,n+/-1) for co*boundaries)
            os.mkdir(config['COMPUTED_VERTEX_FOLDER'])     # create ./sx_path/<name>/computed/0_dfs/ for vertexlist and names and data tagged to it with (index , <) = (node_label, <)
            os.mkdir(config['COMPUTED_EDGE_FOLDER'])       # create ./sx_path/<name>/computed/1_dfs/ for edgelists attached to edgelist_data.csv with 
                                                           # source_ix, target_ix according to vertexlist_index int! but additional data, labels, names welcome 
                                                           # no assumptions on symm, dag, multi, loop, whatever, that will be edgelist_dag.csv, edgelist_symm.csv ... etc                                                           
            return 

        NAMED_ROOT = config_stub.get_key('complex_data_folder') + name + '/'
        CONFIG_FILE_PATH = NAMED_ROOT + 'pathing.csv'
        if name not in os.listdir(config_stub.get_key('complex_data_folder')):
            """
            first time configure all the pathing, if the complex <name> is not already a path 
            in our sx_path-explorer folder
            """
            __set_up_pathing__(NAMED_ROOT)                    
        self.config = config_handler.read_from_full_filename(CONFIG_FILE_PATH)
        
    def set_raw_source_name(self, source_name:str):
        if self.config.is_key_set('target'):
            assert source_name!=self.config.get_key('target'), "source and target column need different labels"
        if not self.config.is_key_set('source'):
            self.config.set_new_key(key='source', value=source_name)
        else:
            self.config.set_update_key(key='source', value=source_name)

    def set_raw_target_name(self, target_name:str):
        if self.config.is_key_set('source'):
            assert target_name!=self.config.get_key('source'), "source and target column need different labels"
        if not self.config.is_key_set('target'):
            self.config.set_new_key(key='target', value=target_name)
        else:
            self.config.set_update_key(key='target', value=target_name)

    @property
    def source(self):
        if self.config.is_key_set('source'):
            return self.config.get_key('source')
        return 'source'

    @property
    def target(self):
        if self.config.is_key_set('target'):
            return self.config.get_key('target')
        return 'target'

    def load_raw_VE_data_from_edgelist(self, edgelist_df:pd.DataFrame, source_name:str='source', target_name:str='target'):
        """
        ONLY EXPECTED TO BE USED ONCE PER MODEL, NO UPDATE YET; JUST WON'T DO A THING
        expect a dataframe like: edgelist_df = <'source','target', [data] >
        compile raw vertex list as sorted(set(edgelist_df.source.unique(), edgelist_df.target.unique()))
            and assign integer labels per index totally ordered like the labels before
        compile a raw edge list with source and target integer indexed by vertex list dict
        """
        if not os.listdir(self.config.get_key('RAW_FOLDER')):
            self.set_raw_source_name(source_name)
            self.set_raw_target_name(target_name)
            edgelist_df = edgelist_df.sort_values([self.source, self.target]).reset_index().drop(columns='index')
            unique_vertex_labels = sorted(set(list(edgelist_df[self.source].unique()) + list(edgelist_df[self.target].unique())))
            vertex_index_assignment_dict = dict(enumerate(unique_vertex_labels))
            vertex_indices = vertex_index_assignment_dict.keys()
            #save map  V: [n] -> Labels as key-value-dict-csv, bijection along total label-order per two lines before
            config_handler.to_key_value(d=vertex_index_assignment_dict, filename=self.config.get_key('RAW_FOLDER_PLUS_RAW_INFIX')+'vertices.csv')
            #make map V^-1: Labels -> [n] to relabel 
            vertex_label_to_index_dict = config_handler.from_key_value(self.config.get_key('RAW_FOLDER_PLUS_RAW_INFIX')+'vertices.csv', flipped=True)
            pd.DataFrame(vertex_indices, columns=['index_0'])\
                .to_csv(self.config.get_key('COMPUTED_VERTEX_FOLDER')+'0.csv', index=False)
            edgelist_df[self.source] = edgelist_df[self.source].apply(lambda x: vertex_label_to_index_dict[x])
            edgelist_df[self.target] = edgelist_df[self.target].apply(lambda x: vertex_label_to_index_dict[x])
            edge_columns = [self.source, self.target,] 
            edge_columns = edge_columns + list(filter(lambda x: x not in edge_columns, edgelist_df.columns))
            edgelist_df = edgelist_df[edge_columns] 
            #arrange source, target into the first two columns, rest of the data follows, edge index given implicitly by row number
            edgelist_df.to_csv(self.config.get_key('RAW_FOLDER_PLUS_RAW_INFIX')+'edges.csv',index=False)
            edgelist_df[[self.source, self.target]].reset_index().rename(columns={'index':'index_1'})\
                .to_csv(self.config.get_key('COMPUTED_EDGE_FOLDER')+'1_data_order.csv', index=False)

    @property
    def vertex_list(self):
        return list(pd.read_csv(self.config.get_key('COMPUTED_VERTEX_FOLDER')+'0.csv')['index_0'])

    @property
    def data_edgelist(self):
        """
        edgelist as in data
        returned as [(u,v),(x,y),(a,b),...] list
        """    
        tmp_df = pd.read_csv(self.config.get_key('COMPUTED_EDGE_FOLDER')+'1_data_order.csv')
        tmp = list(zip(tmp_df[self.source],tmp_df[self.target]))
        del tmp_df
        return tmp
    
    @property
    def data_out(self):
        edges = self.data_edgelist
        tmp_adj = pd.DataFrame(edges, columns=['s','t']).groupby('s').agg(list).to_dict('index')
        tmp_adj = {k:tmp_adj[k]['t'] for k in tmp_adj}
        return tmp_adj

    @property
    def data_in(self):
        edges = self.data_edgelist
        tmp_adj = pd.DataFrame(edges, columns=['s','t']).groupby('t').agg(list).to_dict('index')
        tmp_adj = {k:tmp_adj[k]['s'] for k in tmp_adj}
        return tmp_adj

    @property
    def data_adj(self):
        in_dict = self.data_in
        out_dict = self.data_out
        tmp = {k: in_dict[k] for k in in_dict}
        tmp = {k: (out_dict[k]+tmp[k] if (k in tmp.keys() and k in out_dict.keys())
                                    else (out_dict[k] if k in out_dict.keys() else tmp[k]))\
                 for k in sorted(set(list(out_dict.keys()) + list(tmp.keys()))) }
        return tmp 

    @property
    def unique_undir_edgelist(self):
        """
        sorted edgelist [(u,v),(x,y),(a,b),...] with either u,v or v,u in data and u<v in index (no loops)
        """
        if '1_unique_undir.csv' not in os.listdir(self.config.get_key('COMPUTED_EDGE_FOLDER')):
            tmp_edgelist = self.data_edgelist.copy()
            tmp_edgelist = tmp_edgelist + list(map(lambda x: (x[1],x[0]), tmp_edgelist))
            tmp_edgelist = sorted(set(filter(lambda x: x[0]<x[1] , tmp_edgelist)))
            df = pd.DataFrame(tmp_edgelist,columns=[self.source, self.target])\
                .reset_index().rename(columns={'index':'index_1_unique_undir'})
            df.to_csv(self.config.get_key('COMPUTED_EDGE_FOLDER')+'1_unique_undir.csv',index=False)
            del df
            return tmp_edgelist
        df = pd.read_csv(self.config.get_key('COMPUTED_EDGE_FOLDER')+'1_unique_undir.csv')
        tmp = sorted(set(zip(df[self.source],df[self.target])))
        del df
        return tmp

    @property 
    def symm_undir_edgelist(self):
        """
        sorted edgelist [(u,v),(v,u),(a,b),(b,a),...] with either u,v or v,u in data (no loops)
        """
        if '1_symm_undir.csv' not in os.listdir(self.config.get_key('COMPUTED_EDGE_FOLDER')):
            tmp = sorted(self.unique_undir_edgelist + list(map(lambda x: (x[1],x[0]), self.unique_undir_edgelist)))
            df = pd.DataFrame(tmp,columns=[self.source, self.target])
            df.to_csv(self.config.get_key('COMPUTED_EDGE_FOLDER')+'1_symm_undir.csv')
            del df 
        else:        
            df = pd.read_csv(self.config.get_key('COMPUTED_EDGE_FOLDER')+'1_symm_undir.csv')
            tmp = sorted(list(zip(df[self.source],df[self.target])))
            del df 
        return tmp 

    @property
    def data_dfs_tree(self):
        """
        get a dfs tree according to the data-native edges
        """
        return 

cx = complex_explorer_from_graph('elbformat')          
df = pd.read_csv('2021-02-23_elbformat.de_graph.csv').rename(columns={'from':'source', 'to':'target'})
cx.load_raw_VE_data_from_edgelist(edgelist_df=df)

def simple_short_paths(df, lim:int=10)->dict:
    def iterated_join_edgelist(df:pd.DataFrame, lim:int=20)->pd.DataFrame:
        def clean_last_vertex_to_simple_path(l):
            l = list(l)
            if l[-1] in l[:-1]:
                return -1
            return l[-1]
        def paths_postprocess(path_df:pd.DataFrame)->pd.DataFrame:
            path_df = pd.DataFrame(path_df.apply(tuple,axis=1).apply(lambda x: tuple(filter(lambda z: z>-1,x))), columns=['simple_path_tuple'])
            path_df['v_length'] = path_df.simple_path_tuple.apply(len)
            path_df['p_0'] = path_df.simple_path_tuple.apply(lambda x: x[0])
            path_df['t'] = path_df.simple_path_tuple.apply(lambda x: x[-1])
            path_df = path_df.sort_values(['v_length', 'p_0', 't', 'simple_path_tuple']).reset_index().drop(columns=['index'])
            return path_df
        assert list(df.columns) == [0,1]
        loop_free_edges = df[df[0]!=df[1]].copy()
        tmp_df = loop_free_edges.copy()
        for i in range(2,lim):
            tmp_df = tmp_df.merge(loop_free_edges.rename(columns={0:i-1,1:i}), 'left').fillna(-1).applymap(int)
            tmp_df[i] = tmp_df.apply(clean_last_vertex_to_simple_path,axis=1)
            if tmp_df[i].max() == -1:
                return paths_postprocess(tmp_df.drop(columns=i))
            tmp_df = tmp_df.sort_values(i,ascending=False)
        return paths_postprocess(tmp_df)
    loop_free_edges = df[df[0]!=df[1]].copy()
    path_df = iterated_join_edgelist(loop_free_edges, lim)
    return path_df

ddf = simple_short_paths(pd.DataFrame(cx.data_edgelist))
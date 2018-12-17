try :
    from ConfigParser import ConfigParser 
except ImportError :
    from configparser import ConfigParser 

from collections import OrderedDict
from ast import literal_eval

def config_read(config_fnam, val_doc_adv=False):
    """
    Parse configuration files of the form:
        # comment <-- not parsed
        [group-name]
        key = val ; doc string
        [group-name-advanced]
        key = val ; doc string
    
    In the output dictionary all lines in 'group-name-advanced' are merged with 'group-name'.
    We attempt to parse 'val' as a python object, if this fails then it is parsed as a string. 
    
    Parameters
    ----------
    config_fnam : string or list or strings
        filename of the configuration file/s to be parsed, the first sucessfully parsed file is parsed.
    
    val_doc_adv : bool
        Toggles the output format of 'config_dict', see below.
    
    Returns
    -------
    config_dict : OrderedDict
        If val_doc_adv is True then config_dict is an ordered dictionary of the form:
            output = {group-name: {key : (eval(val), doc_string, is_advanced)}}
        
        Every value in the returned dicionary is a len 3 tuple of the form:
            (val, doc_string, is_advanced)
        
        If the doc string is not suplied in the file then doc_string is None. If 
        val_doc_adv is False then config_dict is an ordered dictionary of the form:
            output = {group-name: {key : eval(val)}}
    
    fnam : str
        A string containing the name of the sucessfully read file.
    """
    c     = ConfigParser()
    fnams = c.read(config_fnam)
    if len(fnams) == 0 :
        raise ValueError('could not find config file: ' + str(config_fnam))

    # read the first sucessfully read file
    c    = ConfigParser()
    fnam = c.read(fnams[0])
    out  = OrderedDict()
    for sect in c.sections():
        s = sect.split('-advanced')[0]
        
        if s not in out :
            out[s] = OrderedDict()
        
        advanced = sect.endswith('-advanced')
        
        for op in c.options(sect):
            # split on ';' to separate doc string from value
            doc  = None
            vals = c.get(sect, op).split(';')
            if len(vals) == 1 :
                v = vals[0].strip()
            elif len(vals) == 2 :
                v   = vals[0].strip()
                doc = vals[1].strip()
            else :
                raise ValueError('could not parse config line' + str(sect) + ' ' + c.get(sect, op))
            
            # try to safely evaluate the str as a python object
            try : 
                v = literal_eval(v)
            except ValueError :
                pass
            except SyntaxError :
                pass
            
            # add to the dictionary 
            out[s][op] = (v, doc, advanced)
    
    if val_doc_adv is False :
        out2 = OrderedDict()
        for s in out.keys():
            out2[s] = OrderedDict()
            for k in out[s].keys():
                out2[s][k] = out[s][k][0]
    else :
        out2 = out
    return out2, fnam[0]

def config_write(con_dict, fnam, val_doc_adv=False):
    """
    """
    def write_adv_item(f, cdict, group, key):
        val, doc, advanced = cdict[group][key]
        out_str = key
        out_str = out_str + ' = '
        out_str = out_str + str(val).strip()
        if doc is not None :
            out_str = out_str + ' ;'
            out_str = out_str + str(doc).strip()
        f.write( out_str + '\n')

    # write 'normal' {'group': {'key' : val}}
    # as config file:
    # [group]
    # key = val
    if val_doc_adv is False :
        with open(fnam, 'w') as f:
            for group in con_dict.keys():
                f.write('['+group+']' + '\n')
                
                for key in con_dict[group].keys():
                    out_str = key
                    out_str = out_str + ' = '
                    out_str = out_str + str(con_dict[group][key]).strip()
                    f.write( out_str + '\n')
    # write 'not normal' {'group': {'key' : (val, doc, advanced)}}
    # as config file:
    # [group]
    # key = val ; doc
    # [group-advanced]
    # key = val ; doc
    else :
        advanced_groups = []
        with open(fnam, 'w') as f:
            for group in con_dict.keys():
                f.write('['+group+']' + '\n')
                
                for key in con_dict[group].keys():
                    advanced = con_dict[group][key][2]
                    if not advanced :
                        write_adv_item(f, con_dict, group, key)
                    else :
                        advanced_groups.append(group)
            
            for group in con_dict.keys():
                if group in advanced_groups :
                    f.write('\n['+group+'-advanced]' + '\n')
                    
                    for key in con_dict[group].keys():
                        advanced = con_dict[group][key][2]
                        if advanced :
                            write_adv_item(f, con_dict, group, key)

def write_h5(h5_file, h5_group, d):
    import h5py
    f = h5py.File(h5_file)
    for key, val in d.items(): 
        key2 = h5_group + '/' + key
        
        if key2 in f :
            del f[key2]
        
        print(key2, type(val))
        f[key2] = val
    f.close()


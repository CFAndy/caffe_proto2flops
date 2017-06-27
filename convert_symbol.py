#
#  Original code was ported from MXNET
#
#
from __future__ import print_function
from google.protobuf import text_format
import argparse
import re
import sys
import pprint
import math
from os import *
caffe_flag = True
verbose = False

try:
    import caffe
    from caffe.proto import caffe_pb2
except ImportError:
    caffe_flag = False
    import caffe_parse.caffe_pb2

 
def readProtoSolverFile(filepath):
    solver_config = ''
    if caffe_flag:
        solver_config = caffe.proto.caffe_pb2.NetParameter()
    else:
        solver_config = caffe_parse.caffe_pb2.NetParameter()
    return readProtoFile(filepath, solver_config)

def readProtoFile(filepath, parser_object):
    file = open(filepath, "r")
    if not file:
        raise self.ProcessException("ERROR (" + filepath + ")!")
    text_format.Merge(str(file.read()), parser_object)
    file.close()
    return parser_object

def dupInputSize(input_size):
    output_size=[0,0,0,0]
    for i in range(0,len(input_size)):
        output_size[i] = input_size[i]
    return output_size

def convFlops(param, input_size, pre_group):
    pad = 0
    if isinstance(param.pad, int):
        pad = param.pad
    else:
        pad = 0 if len(param.pad) == 0 else param.pad[0]

    group = 1 
    if isinstance(param.group, int):
        group = param.group

    stride = 1
    if isinstance(param.stride, int):
        stride = param.stride
    else:
        stride = 1 if len(param.stride) == 0 else param.stride[0]

    kernel_size = ''
    if isinstance(param.kernel_size, int):
        kernel_size = param.kernel_size
    else:
        kernel_size = param.kernel_size[0]

    dilate = 1
    if isinstance(param.dilation, int):
        dilate = param.dilation
    else:
        dilate = 1 if len(param.dilation) == 0 else param.dilation[0]

    if pre_group  != group:
        input_size[1] = input_size[1]  * pre_group
        input_size[1] /= group
    # convert to string except for dilation
    param_string = "num_filter=%d, pad=(%d,%d), kernel=(%d,%d)," \
                   " stride=(%d,%d), no_bias=%s, num_intput_channel=%d, group=%d" %\
        (param.num_output, pad, pad, kernel_size,\
        kernel_size, stride, stride, not param.bias_term, input_size[1], group)
    if verbose :  print(param_string)
    # deal with dilation. Won't be in deconvolution
    if dilate > 1:
        param_string += ", dilate=(%d, %d)" % (dilate, dilate)
        

    output_size = [0, 0, 0 , 0]
    output_size[0] = input_size[0]
    #print pre_group, ",", param.group
    output_size[1] = param.num_output / group
    output_size[2] = output_size[3] =  ( input_size[2] + 2 * pad  - kernel_size ) / stride + 1
    flops = 1.0 * output_size[2] * output_size[3] * \
            ( kernel_size * kernel_size  * ( input_size[1] ) )*\
            output_size[1] * output_size[0]

    weight =  ((kernel_size * kernel_size  * ( input_size[1] ) + 1) *  output_size[1] )
    if verbose:   print(input_size, ":", kernel_size, input_size[1], output_size[1])
    return output_size, flops * group, group, weight*group

def proto2flops(proto_file):
    proto = readProtoSolverFile(proto_file)
    connection = dict()
    symbols = dict()
    top = dict()
    flatten_count = 0
    symbol_string = ""
    layer = ''
    if len(proto.layer):
        layer = proto.layer
    elif len(proto.layers):
        layer = proto.layers
    else:
        raise Exception('Invalid proto file.')   
    # Get input size to network
    input_dim = None
    #input_dim = [10, 3, 224, 224] # default
    #input_dim = [10, 3, 192, 192] # default
    #if len(proto.input_dim) > 0:
    #    input_dim = proto.input_dim
    #elif len(proto.input_shape) > 0: 
    #    input_dim = proto.input_shape[0].dim
    # We assume the first bottom blob of first layer is the output from data layer
    #pprint.pprint(layer)
    #pprint.pprint(layer[0])
    #input_name = layer[0].bottom[0]
    input_name = proto.name
    if verbose:  print( "input_name is " + input_name )
    if verbose:  print( "layer, type, input_size, output_size, flops")
    #input_name = "test"
    output_name = ""
    mapping = {input_name : 'data'}
    output_mapping = {}
    need_flatten = {input_name : False}
    gflops = 0.0;
    weight = 0.0;
    gpooling_flops = 0;
    pre_group = 1;
    conv_layer_num = 0;
    for i in range(len(layer)):
        type_string = ''
        param_string = ''
        name = re.sub('[-/]', '_', layer[i].name)
        #print name,",",layer[i].type

        layer_output_size = [0,0,0,0]
        if i != 0 :
            bottom_name = re.sub('[-/]', '_', layer[i].bottom[0])
            layer_input_size  = output_mapping[bottom_name]
            #print "***",",", layer_input_size
            layer_output_size = dupInputSize(layer_input_size)
            output_mapping[name] = layer_output_size
        if layer[i].type == 'Input' :
            type_string = "mx.symbol.Input"
            dim = layer[i].input_param.shape[0].dim
            if input_dim != None:
                dim = input_dim
            if verbose:  print(type(dim))
            output_mapping[name] = dim
            input_dim = dim
            need_flatten[name] = False
        if layer[i].type == 'Convolution' or layer[i].type == 4:
            type_string = 'mx.symbol.Convolution'
            #print name,",",layer[i].type
            layer_output_size, flop, pre_group, conv_weight = convFlops(layer[i].convolution_param, layer_input_size, pre_group)
            output_mapping[name] = layer_output_size
            conv_layer_num = conv_layer_num + 1
            if verbose:  print( conv_layer_num,",",name, "=>", output_mapping[name])
            gflops += flop
            weight += conv_weight
            if verbose:  print( name,",",layer[i].type ,",",layer_input_size,",", layer_output_size,",", flop,",",pre_group, ",", conv_weight)

            need_flatten[name] = True
        if layer[i].type == 'Deconvolution' or layer[i].type == 39:
            type_string = 'mx.symbol.Deconvolution'
            param_string = convFlops(layer[i].convolution_param)
            need_flatten[name] = True
        if layer[i].type == 'Pooling' or layer[i].type == 17:
            type_string = 'mx.symbol.Pooling'
            param = layer[i].pooling_param
            param_string = ''
            
            if param.global_pooling == True:
                #print layer[i]
                #raise Exception("Unknow global pooling " + layer[i].type)
			    layer_output_size[3] = layer_output_size[2] = 1
            else:
                param_string += "pad=(%d,%d), kernel=(%d,%d), stride=(%d,%d)" %\
                    (param.pad, param.pad, param.kernel_size,\
                    param.kernel_size, param.stride, param.stride)
                layer_output_size[3] = layer_output_size[2] = int(math.ceil((layer_input_size[2] + 2 * param.pad - param.kernel_size)*1.0/param.stride) + 1)
            output_mapping[name] = layer_output_size

            #print layer_output_size,"pooling output"
            if param.pool == 0:
                param_string = param_string + ", pool_type='max'"
            elif param.pool == 1:
                param_string = param_string + ", pool_type='avg'"
            else:
                raise Exception("Unknown Pooling Method!")
            need_flatten[name] = True
        if layer[i].type == 'BatchNorm' :
            type_string = 'mx.symbol.BatchNorm'
            param_string = "act_type='BatchNorm'"
            need_flatten[name] = need_flatten[mapping[layer[i].bottom[0]]]
        if layer[i].type == 'ReLU' or layer[i].type == 18:
            type_string = 'mx.symbol.Activation'
            param_string = "act_type='relu'"
            need_flatten[name] = need_flatten[mapping[layer[i].bottom[0]]]
        if layer[i].type == 'LRN' or layer[i].type == 15:
            type_string = 'mx.symbol.LRN'
            param = layer[i].lrn_param
            param_string = "alpha=%f, beta=%f, knorm=%f, nsize=%d" %\
                (param.alpha, param.beta, param.k, param.local_size)
            need_flatten[name] = True
        if layer[i].type == 'InnerProduct' or layer[i].type == 14:
            type_string = 'mx.symbol.FullyConnected'
            #print bottom_name
            param = layer[i].inner_product_param
            param_string = "num_hidden=%d, no_bias=%s" % (param.num_output, not param.bias_term)
            layer_output_size=[0,0]
            layer_output_size[0] = layer_input_size[0]
            layer_output_size[1] = layer[i].inner_product_param.num_output
            layer_input_size[1] *= pre_group
            fc_weight = layer_output_size[1] * ( layer_input_size[1] + 1)
            weight += fc_weight
            #print layer_input_size
            if len(layer_input_size) == 4 :
                flops  =  layer_output_size[0] * (layer_output_size[1] * layer_input_size[1] * layer_input_size[2]* layer_input_size[3]  )
                fc_weight = layer_output_size[1] * (layer_input_size[1] * layer_input_size[2]* layer_input_size[3] + 1)
            else:
                flops  =  layer_output_size[0] * (layer_output_size[1] * layer_input_size[1]  )
            pre_group = 1
            if verbose:  print( name,",",layer[i].type ,",",layer_input_size,",", layer_output_size,",", flops, "," , fc_weight)
            gflops += flops
            output_mapping[name] = layer_output_size
            #print layer_output_size, flops
            need_flatten[name] = False
        if layer[i].type == 'Dropout' or layer[i].type == 6:
            type_string = 'mx.symbol.Dropout'
            #print layer_input_size,",","drop input"
            param = layer[i].dropout_param
            param_string = "p=%f" % param.dropout_ratio
            need_flatten[name] = need_flatten[mapping[layer[i].bottom[0]]]
            #print layer_output_size,",","drop output"
        if layer[i].type == 'Softmax' or layer[i].type == 20:
            type_string = 'mx.symbol.SoftmaxOutput'
        if layer[i].type == 'Flatten' or layer[i].type == 8:
            type_string = 'mx.symbol.Flatten'
            need_flatten[name] = False
        if layer[i].type == 'Split' or layer[i].type == 22:
            type_string = 'split'
        if layer[i].type == 'Concat' or layer[i].type == 3:
            type_string = 'mx.symbol.Concat'
            layer_output_size = [0, 0, 0, 0]
            for x in layer[i].bottom:
                bottom_name = re.sub('[-/]', '_', x)
                sub_layer_input_size = output_mapping[bottom_name]
                #print bottom_name,"?,", sub_layer_input_size
                for idx in range (0,4):
                    if idx == 1 : 
                        layer_output_size[idx] += sub_layer_input_size [idx]
                    else: 
                        layer_output_size[idx] = sub_layer_input_size [idx]
            #print name,",",layer_output_size
            output_mapping[name] = layer_output_size
            need_flatten[name] = True
        if layer[i].type == 'Crop':
            type_string = 'mx.symbol.Crop'
            need_flatten[name] = True
            param_string = 'center_crop=True'
        if layer[i].type == 'BatchNorm':
            type_string = 'mx.symbol.BatchNorm'
            param = layer[i].batch_norm_param
            param_string = 'use_global_stats=%s' % param.use_global_stats
        if type_string == '':
            raise Exception('Unknown Layer %s!' % layer[i].type)
        if type_string != 'split':
            bottom = layer[i].bottom
            if param_string != "":
                param_string = ", " + param_string
            if len(bottom) == 1:
                if need_flatten[mapping[bottom[0]]] and type_string == 'mx.symbol.FullyConnected':
                    flatten_name = "flatten_%d" % flatten_count
                    symbol_string += "%s=mx.symbol.Flatten(name='%s', data=%s)\n" %\
                        (flatten_name, flatten_name, mapping[bottom[0]])
                    flatten_count += 1
                    need_flatten[flatten_name] = False
                    bottom[0] = flatten_name
                    mapping[bottom[0]] = bottom[0]
                symbol_string += "%s = %s(name='%s', data=%s %s)\n" %\
                    (name, type_string, name, mapping[bottom[0]], param_string)
            else:
                symbol_string += "%s = %s(name='%s', *[%s] %s)\n" %\
                    (name, type_string, name, ','.join([mapping[x] for x in bottom]), param_string)
        for j in range(len(layer[i].top)):
            mapping[layer[i].top[j]] = name
        output_name = name
    return input_name, gflops /input_dim[0], weight



def main():
    os.listdir(sys.argv[1])
    #name, mad, weight = proto2flops(sys.argv[1])
    #if verbose: print( "total_conv_and_fc_flops in MillionMAD = %f, MillionParam = %f " % (mad/1e6, weight/1e6))
    #print("%s,%f,%f " % (name, mad / 1e6, weight / 1e6))

if __name__ == '__main__':
    main()

import onnx
import argparse
import numpy as np
from onnxsim import simplify
from collections import OrderedDict
import onnx_graphsurgeon as gs

def addLayerNormPlugin(sourceOnnx,destinationOnnx):
    bLayerNormPlugin = True
    nLayerNormPlugin = 0
    graph = gs.import_onnx(onnx.shape_inference.infer_shapes(onnx.load(sourceOnnx)))

    if bLayerNormPlugin:
        for node in graph.nodes:
            if node.op == 'ReduceMean' and \
                node.o().op == 'Sub' and node.o().inputs[0] == node.inputs[0] and \
                node.o().o(0).op =='Pow' and node.o().o(1).op =='Div' and \
                node.o().o(0).o().op == 'ReduceMean' and \
                node.o().o(0).o().o().op == 'Add' and \
                node.o().o(0).o().o().o().op == 'Sqrt' and \
                node.o().o(0).o().o().o().o().op == 'Div' and node.o().o(0).o().o().o().o() == node.o().o(1):

                inputTensor = node.inputs[0]
                lastDivNode = node.o().o(0).o().o().o().o()

                if lastDivNode.o().op=='Mul' and lastDivNode.o().o().op=='Add':
                    weight=lastDivNode.o().inputs[1]
                    bias=lastDivNode.o().o().inputs[1]
                    layerNormN = gs.Node("LayerNorm", "LayerNorm-" + str(nLayerNormPlugin), inputs= [inputTensor], outputs=lastDivNode.o().o().outputs)
                    layerNormN.attrs = OrderedDict([("weight", weight),("bias", bias)])  #
                    graph.nodes.append(layerNormN)
                    print("LayerNorm-" + str(nLayerNormPlugin))
                    nLayerNormPlugin += 1
                    lastDivNode.o().o().outputs = []
                continue

    graph.cleanup()
    onnx.save(gs.export_onnx(graph), destinationOnnx)
def addAttentionPlugin(sourceOnnx,destinationOnnx):
     graph = gs.import_onnx(onnx.shape_inference.infer_shapes(onnx.load(sourceOnnx)))
     nlayer=0
     for node in graph.nodes:
            if node.op == 'Split' and node.o().op=='MatMul' and node.o().o().op=='Mul':#
                #print(node.o(0).op,node.o().o().op)
                inputs=node.inputs
                outputs=node.o().o().o().o().outputs
                scale=node.o().o().inputs[1]
                AttentionN = gs.Node("Attention", "Attention-" + str(nlayer), inputs= inputs, outputs=outputs)
                AttentionN.attrs = OrderedDict([("scale", scale)])
                graph.nodes.append(AttentionN)
                print("Attention-" + str(nlayer))
                node.o().o().o().o().outputs=[]
                nlayer +=1
     graph.cleanup()
     onnx.save(gs.export_onnx(graph), destinationOnnx)    

if __name__=="__main__":
  parser = argparse.ArgumentParser(description='optimize onnx describe.')
  parser.add_argument(
      "--input_path",
      type = str,
      default="./target/MobileViT.onnx",
      help="input onnx model path, default is ./target/MobileViT.onnx.")

  parser.add_argument(
      "--save_path",
      type=str,
      default="./target/MobileViT_final.onnx",
      help="save direction of onnx models,default is ./target.")

  args = parser.parse_args()
  print(args)
  
  addLayerNormPlugin(args.input_path,args.save_path)
  addAttentionPlugin(args.save_path,args.save_path)
  

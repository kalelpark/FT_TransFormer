# --- Just Revisiting Deep Learning Models for Tabular Data

# Argument Explanation
# --action : train or test [but, if you need to for review, using train!(This can give all about information!)]
# --model : Choose FT Transformer, ResNet
# --datapath : Choose data [check data file or README.md]
# --savepath : saved all about Information

# If you any Question, leave an Issue.

# TIP!
# FT-Transformer

# Regression
# python main.py --action train --model fttransformer --data california_housing --savepath output         # 0.01
# python main.py --action train --model fttransformer --data microsoft --savepath output                  # 0.03
# python main.py --action train --model fttransformer --data yahoo --savepath output                      
# python main.py --action train --model fttransformer --data year --savepath output 

# Multi classification
# python main.py --action train --model fttransformer --data aloi --savepath output                        # 0.06
# python main.py --action train --model fttransformer --data covtype --savepath output                     # 0.005
# python main.py --action train --model fttransformer --data helena --savepath output                      # 0.016
# python main.py --action train --model fttransformer --data jannis --savepath output

# Binary Classification
# python main.py --action train --model fttransformer --data epsilon --savepath output                   
# python main.py --action train --model fttransformer --data higgs_small --savepath output                  


# ResNet

# Regression
# python main.py --action train --model resnet --data california_housing --savepath output         +- 0.01
# python main.py --action train --model resnet --data microsoft --savepath output
# python main.py --action train --model resnet --data yahoo --savepath output
# python main.py --action train --model resnet --data year --savepath output 

# Multi classification
# python main.py --action train --model resnet --data aloi --savepath output                      
# python main.py --action train --model resnet --data covtype --savepath output
# python main.py --action train --model resnet --data helena --savepath output
# python main.py --action train --model resnet --data jannis --savepath output

# Binary Classification
# python main.py --action train --model resnet --data epsilon --savepath output                   
# python main.py --action train --model resnet --data higgs_small --savepath output  
# --- Just Revisiting Deep Learning Models for Tabular Data

# Argument Explanation
# --action : train or test [but, if you need to for review, using train!(This can give all about information!)]
# --model : Choose FT Transformer, ResNet
# --datapath : Choose data [check data file or README.md]
# --savepath : saved all about Information

# If you any Question, leave an Issue.

# TIP!
# FT-Transformer
# python main.py --action train --model fttransformer --data california_housing --savepath output        // Regression -> Clear
# python main.py --action train --model fttransformer --data aloi --savepath output                      // Multi Label 
# python main.py --action train --model fttransformer --data epsilon --savepath output                  // Binary Label

# ResNet
# python main.py --action train --model resnet --data california_housing --savepath output               // Regression
# python main.py --action train --model resnet --data aloi --savepath output               // Multi Label
# python main.py --action train --model resnet --data epsilon --savepath output               // Binary Label

# XGBoost (Later Update!)
# python main.py --action train --model resnet --datapath data/microsoft --savepath output/
# python main.py --action train --model resnet --data microsoft --savepath output
####
 # Filename:        listGen.sh
 # Date:            Sep 20 2016
 # Last Edited by:  Gengshan Yang
 # Description:     Usage: ./listGen.sh path-to-pos-train path-to-neg-train
 #                         path-to-pos-val path-to-neg-val
 ####

currPath=$PWD 

cd $1
ls -d -1 $PWD/*.* > ${currPath}/data/listPosTrain.txt

cd $2
ls -d -1 $PWD/*.* > ${currPath}/data/listNegTrain.txt

cd $3
ls -d -1 $PWD/*.* > ${currPath}/data/listPosVal.txt

cd $4
ls -d -1 $PWD/*.* > ${currPath}/data/listNegVal.txt

cd $currPath

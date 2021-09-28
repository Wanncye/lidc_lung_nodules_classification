import json
import pandas as pd
import numpy as np
import random

modelList = ['alexnet','attention56','vgg13','resnet34']
for i in range(1):
    ensembleMeanList = []
    for fold in range(5):
        for model in modelList:
            jsonFileName = 'folder.'+str(fold)+'.FocalLoss_alpha_0.25.metrics_val_best_weights_.json'
            jsonFilePath = 'experiments/'+model+'_nomask/'+jsonFileName
            f = open(jsonFilePath,'r')
            jsonData = json.load(f)
            jsonAcc = jsonData['accuracy']
            jsonEpoch = jsonData['epoch']
            print("{2}_fold_{0}_epoch_{1}".format(fold,jsonEpoch-1,model))
            csvPath = 'experiments/'+\
                        model+\
                        '_nomask/result_<=20mm_nodule_gcn_traditional_addEightLabelFeature_norInput_testZero/'+\
                        'folder_'+\
                        str(fold)+\
                        '_result_'+\
                        str(int(jsonEpoch-1))+\
                        '.csv'
            csvReader = pd.read_csv(csvPath)
            groundTruth = csvReader['truth_label']
            filename = csvReader['filename']
            if model=='alexnet':
                alexnetPredLabel = csvReader['predict_label']
                alexnetgroundTruth = csvReader['truth_label']
                alexnetIsRight = csvReader['is_right']
                alexnetWrong = set(filename[alexnetIsRight==False])
                alexnetJsonAcc = jsonAcc
                alexnetCaculAcc = np.sum(groundTruth == alexnetPredLabel)/len(groundTruth)
            if model=='attention56':
                attention56PredLabel = csvReader['predict_label']
                attention56groundTruth = csvReader['truth_label']
                attention56IsRight = csvReader['is_right']
                attention56Wrong = set(filename[attention56IsRight==False])
                attention56JsonAcc = jsonAcc
                attention56CaculAcc = np.sum(groundTruth == attention56PredLabel)/len(groundTruth)
            if model=='vgg13':
                vgg13PredLabel = csvReader['predict_label']
                vgg13groundTruth = csvReader['truth_label']
                vgg13IsRight = csvReader['is_right']
                vgg13Wrong = set(filename[vgg13IsRight==False])
                vgg13JsonAcc = jsonAcc
                vgg13CaculAcc = np.sum(groundTruth == vgg13PredLabel)/len(groundTruth)
            if model=='resnet34':
                resnet34PredLabel = csvReader['predict_label']
                resnet34groundTruth = csvReader['truth_label']
                resnet34IsRight = csvReader['is_right']
                resnet34Wrong = set(filename[resnet34IsRight==False])
                resnet34JsonAcc = jsonAcc
                resnet34CaculAcc = np.sum(groundTruth == resnet34PredLabel)/len(groundTruth)
        print("四个模型全部分错的结节：{0}".format(alexnetWrong & attention56Wrong & vgg13Wrong & resnet34Wrong))
        accSum = alexnetCaculAcc + attention56CaculAcc + vgg13CaculAcc + resnet34CaculAcc
        modelWeight = np.array([alexnetCaculAcc,attention56CaculAcc,vgg13CaculAcc,resnet34CaculAcc])/accSum
        print('modelWeight:',modelWeight)

        
        alexnetPredLabel = modelWeight[0] * alexnetPredLabel
        vgg13PredLabel = modelWeight[1] * vgg13PredLabel
        resnet34PredLabel = modelWeight[2] * resnet34PredLabel
        attention56PredLabel = modelWeight[3] * attention56PredLabel
        finalPredLabel = np.array(alexnetPredLabel + vgg13PredLabel + resnet34PredLabel + attention56PredLabel)
        finalPredLabel = np.where(finalPredLabel>0.5, 1, 0)
        # print('finalPredLabel:',finalPredLabel)
        ensembleAcc = np.sum(groundTruth == finalPredLabel)/len(groundTruth)
        ensembleMeanList.append(ensembleAcc)
        print('alexnet acc: {0}, vgg13 acc: {1}, resnet34 acc: {2}, attention56 acc: {3}, ensemble acc: {4}'.format(
            alexnetCaculAcc,
            vgg13CaculAcc,
            resnet34CaculAcc,
            attention56CaculAcc,
            ensembleAcc,
        ))
        print()
    print('final mean acc: ',np.mean(ensembleMeanList))
    print()

        
        
            
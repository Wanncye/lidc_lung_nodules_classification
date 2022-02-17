import json
import pandas as pd
import numpy as np
import random

modelList = ['alexnet','attention56','vgg13','resnet34']
for i in range(1):
    ensembleMeanList = []
    for fold in range(5):
        for model in modelList:
            jsonFileName = 'folder.'+str(fold)+'.FocalLoss_alpha_0.25_<=20mm_nodule_gcn_traditional_addEightLabelFeature_norInput_testZero.metrics_val_last_weights.json'
            jsonFilePath = 'experiments/'+model+'_nomask/'+jsonFileName
            f = open(jsonFilePath,'r')
            jsonData = json.load(f)
            jsonAcc = jsonData['accuracy']
            jsonEpoch = jsonData['epoch']
            csvPath = 'experiments/'+\
                        model+\
                        '_nomask/result_<=20mm_nodule_gcn_traditional_fold4_addEightLabelFeature_norInput/'+\
                        'folder_'+\
                        str(fold)+\
                        '_result_'+\
                        str(int(jsonEpoch-1))+\
                        '.csv'
            csvReader = pd.read_csv(csvPath)
            groundTruth = csvReader['truth_label']
            if model=='alexnet':
                alexnetPredLabel = csvReader['predict_label']
                alexnetJsonAcc = jsonAcc
                alexnetCaculAcc = np.sum(groundTruth == alexnetPredLabel)/len(groundTruth)
            if model=='attention56':
                attention56PredLabel = csvReader['predict_label']
                attention56JsonAcc = jsonAcc
                attention56CaculAcc = np.sum(groundTruth == attention56PredLabel)/len(groundTruth)
            if model=='vgg13':
                vgg13PredLabel = csvReader['predict_label']
                vgg13JsonAcc = jsonAcc
                vgg13CaculAcc = np.sum(groundTruth == vgg13PredLabel)/len(groundTruth)
            if model=='resnet34':
                resnet34PredLabel = csvReader['predict_label']
                resnet34JsonAcc = jsonAcc
                resnet34CaculAcc = np.sum(groundTruth == resnet34PredLabel)/len(groundTruth)
        # alexnetCaculAcc = random.random()
        # attention56CaculAcc = random.random()
        # vgg13CaculAcc = random.random()
        # resnet34CaculAcc = random.random()
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
    print('final mean acc: ',np.mean(ensembleMeanList))
    print()

        
        
            
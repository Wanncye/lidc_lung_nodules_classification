import json
import pandas as pd
import numpy as np
import random
from sklearn.metrics import confusion_matrix

modelList = ['alexnet','attention56','vgg13','resnet34','googlenet']
descripe = 'para1_10fold_noNorm_add_traditional'

for i in range(1):
    alexnetMeanList = []
    alexnetSensList = []
    alexnetSpecList = []
    alexnetF1List = []
    alexnetPrecList = []
    vggMeanList = []
    vggSensList = []
    vggSpecList = []
    vggF1List = []
    vggPrecList = []
    attentionMeanList = []
    attentionSensList = []
    attentionSpecList = []
    attentionF1List = []
    attentionPrecList = []
    resnetMeanList = []
    resnetSensList = []
    resnetSpecList = []
    resnetF1List = []
    resnetPrecList = []
    googlenetMeanList = []
    googleSensList = []
    googleSpecList = []
    googleF1List = []
    googlePrecList = []
    ensembleMeanList = []
    ensembleSensList = []
    ensembleSpecList = []
    ensembleF1List = []
    ensemblePrecList = []
    for fold in range(10):
        for model in modelList:
            jsonFileName = 'folder.'+str(fold)+'.FocalLoss_alpha_0.25_'+descripe+'.metrics_val_best_weights.json'
            jsonFilePath = 'experiments/'+model+'_nomask/'+jsonFileName
            f = open(jsonFilePath,'r')
            jsonData = json.load(f)
            jsonAcc = jsonData['accuracy']
            jsonEpoch = jsonData['epoch']
            print("{2}_fold_{0}_epoch_{1}".format(fold,jsonEpoch-1,model))
            csvPath = 'experiments/'+\
                        model+\
                        '_nomask/result_'+\
                        descripe +\
                        '/folder_'+\
                        str(fold)+\
                        '_result_'+\
                        str(int(jsonEpoch-1))+\
                        '.csv'
            csvReader = pd.read_csv(csvPath)
            # print(csvPath)
            groundTruth = csvReader['truth_label']
            filename = csvReader['filename']
            if model=='alexnet':
                alexnetPredLabel = csvReader['predict_label']
                alexnetgroundTruth = csvReader['truth_label']
                alexnetIsRight = csvReader['is_right']
                alexnetprobability = csvReader['probability']
                alexnetSoftmax = np.zeros((len(alexnetprobability),2))
                for predLabelIndex,predLabel in enumerate(alexnetPredLabel):
                    if predLabel == 0:
                        alexnetSoftmax[predLabelIndex,0] =  alexnetprobability[predLabelIndex]
                        alexnetSoftmax[predLabelIndex,1] =  1-alexnetprobability[predLabelIndex]
                    if predLabel == 1:
                        alexnetSoftmax[predLabelIndex,1] =  alexnetprobability[predLabelIndex]
                        alexnetSoftmax[predLabelIndex,0] =  1-alexnetprobability[predLabelIndex]

                alexnetWrong = set(filename[alexnetIsRight==False])
                alexnetJsonAcc = jsonAcc
                alexnetCaculAcc = np.sum(groundTruth == alexnetPredLabel)/len(groundTruth)
                con_matrix = confusion_matrix(groundTruth,alexnetPredLabel,labels=range(2))
                alexnetSensitivity = con_matrix[1,1]/(con_matrix[1,0]+con_matrix[1,1])
                alexnetSpecificity = con_matrix[0,0]/(con_matrix[0,0]+con_matrix[0,1])
                alexnetPrecision = con_matrix[0,0]/(con_matrix[0,0]+con_matrix[1,0])
                alexnetF1Score = 2*(alexnetPrecision*alexnetSensitivity)/(alexnetPrecision+alexnetSensitivity)
                print('accuracy:{2},sensitivity:{0},specificity:{1},precision:{4},F1Score:{3}'.format(alexnetSensitivity,alexnetSpecificity,alexnetCaculAcc,alexnetF1Score,alexnetPrecision))
                alexnetSensList.append(alexnetSensitivity)
                alexnetSpecList.append(alexnetSpecificity)
                alexnetF1List.append(alexnetF1Score)
                alexnetPrecList.append(alexnetPrecision)
            if model=='attention56':
                attention56PredLabel = csvReader['predict_label']
                attention56groundTruth = csvReader['truth_label']
                attention56IsRight = csvReader['is_right']
                attention56probability = csvReader['probability']
                attention56Softmax = np.zeros((len(attention56probability),2))
                for predLabelIndex,predLabel in enumerate(attention56PredLabel):
                    if predLabel == 0:
                        attention56Softmax[predLabelIndex,0] =  attention56probability[predLabelIndex]
                        attention56Softmax[predLabelIndex,1] =  1-attention56probability[predLabelIndex]
                    if predLabel == 1:
                        attention56Softmax[predLabelIndex,1] =  attention56probability[predLabelIndex]
                        attention56Softmax[predLabelIndex,0] =  1-attention56probability[predLabelIndex]

                attention56Wrong = set(filename[attention56IsRight==False])
                attention56JsonAcc = jsonAcc
                attention56CaculAcc = np.sum(groundTruth == attention56PredLabel)/len(groundTruth)
                con_matrix = confusion_matrix(groundTruth,attention56PredLabel,labels=range(2))
                attentionSensitivity = con_matrix[1,1]/(con_matrix[1,0]+con_matrix[1,1])
                attentionSpecificity = con_matrix[0,0]/(con_matrix[0,0]+con_matrix[0,1])
                attentionPrecision = con_matrix[0,0]/(con_matrix[0,0]+con_matrix[1,0])
                attentionF1Score = 2*(attentionPrecision*attentionSensitivity)/(attentionPrecision+attentionSensitivity)
                print('accuracy:{2},sensitivity:{0},specificity:{1},precision:{4},F1Score:{3}'.format(attentionSensitivity,attentionSpecificity,attention56CaculAcc,attentionF1Score,attentionPrecision))
                attentionSensList.append(attentionSensitivity)
                attentionSpecList.append(attentionSpecificity)
                attentionF1List.append(attentionF1Score)
                attentionPrecList.append(attentionPrecision)
            if model=='vgg13':
                vgg13PredLabel = csvReader['predict_label']
                vgg13groundTruth = csvReader['truth_label']
                vgg13IsRight = csvReader['is_right']
                vgg13probability = csvReader['probability']
                vgg13Softmax = np.zeros((len(vgg13probability),2))
                for predLabelIndex,predLabel in enumerate(vgg13PredLabel):
                    if predLabel == 0:
                        vgg13Softmax[predLabelIndex,0] =  vgg13probability[predLabelIndex]
                        vgg13Softmax[predLabelIndex,1] =  1-vgg13probability[predLabelIndex]
                    if predLabel == 1:
                        vgg13Softmax[predLabelIndex,1] =  vgg13probability[predLabelIndex]
                        vgg13Softmax[predLabelIndex,0] =  1-vgg13probability[predLabelIndex]

                vgg13Wrong = set(filename[vgg13IsRight==False])
                vgg13JsonAcc = jsonAcc
                vgg13CaculAcc = np.sum(groundTruth == vgg13PredLabel)/len(groundTruth)
                con_matrix = confusion_matrix(groundTruth,vgg13PredLabel,labels=range(2))
                vgg13Sensitivity = con_matrix[1,1]/(con_matrix[1,0]+con_matrix[1,1])
                vgg13Specificity = con_matrix[0,0]/(con_matrix[0,0]+con_matrix[0,1])
                vgg13Precision = con_matrix[0,0]/(con_matrix[0,0]+con_matrix[1,0])
                vgg13F1Score = 2*(vgg13Precision*vgg13Sensitivity)/(vgg13Precision+vgg13Sensitivity)
                print('accuracy:{2},sensitivity:{0},specificity:{1},precision:{4},F1Score:{3}'.format(vgg13Sensitivity,vgg13Specificity,vgg13CaculAcc,vgg13F1Score,vgg13Precision))
                vggSensList.append(vgg13Sensitivity)
                vggSpecList.append(vgg13Specificity)
                vggF1List.append(vgg13F1Score)
                vggPrecList.append(vgg13Precision)
            if model=='resnet34':
                resnet34PredLabel = csvReader['predict_label']
                resnet34groundTruth = csvReader['truth_label']
                resnet34IsRight = csvReader['is_right']
                resnet34probability = csvReader['probability']
                resnet34Softmax = np.zeros((len(resnet34probability),2))
                for predLabelIndex,predLabel in enumerate(resnet34PredLabel):
                    if predLabel == 0:
                        resnet34Softmax[predLabelIndex,0] =  resnet34probability[predLabelIndex]
                        resnet34Softmax[predLabelIndex,1] =  1-resnet34probability[predLabelIndex]
                    if predLabel == 1:
                        resnet34Softmax[predLabelIndex,1] =  resnet34probability[predLabelIndex]
                        resnet34Softmax[predLabelIndex,0] =  1-resnet34probability[predLabelIndex]

                resnet34Wrong = set(filename[resnet34IsRight==False])
                resnet34JsonAcc = jsonAcc
                resnet34CaculAcc = np.sum(groundTruth == resnet34PredLabel)/len(groundTruth)
                con_matrix = confusion_matrix(groundTruth,resnet34PredLabel,labels=range(2))
                resnet34Sensitivity = con_matrix[1,1]/(con_matrix[1,0]+con_matrix[1,1])
                resnet34Specificity = con_matrix[0,0]/(con_matrix[0,0]+con_matrix[0,1])
                print('accuracy:{2},sensitivity:{0},specificity:{1}'.format(resnet34Sensitivity,resnet34Specificity,resnet34CaculAcc))
                resnetSensList.append(resnet34Sensitivity)
                resnetSpecList.append(resnet34Specificity)
            if model=='googlenet':
                googlenetPredLabel = csvReader['predict_label']
                googlenetgroundTruth = csvReader['truth_label']
                googlenetIsRight = csvReader['is_right']
                googlenetprobability = csvReader['probability']
                googlenetSoftmax = np.zeros((len(googlenetprobability),2))
                for predLabelIndex,predLabel in enumerate(googlenetPredLabel):
                    if predLabel == 0:
                        googlenetSoftmax[predLabelIndex,0] =  googlenetprobability[predLabelIndex]
                        googlenetSoftmax[predLabelIndex,1] =  1-googlenetprobability[predLabelIndex]
                    if predLabel == 1:
                        googlenetSoftmax[predLabelIndex,1] =  googlenetprobability[predLabelIndex]
                        googlenetSoftmax[predLabelIndex,0] =  1-googlenetprobability[predLabelIndex]

                googlenetWrong = set(filename[googlenetIsRight==False])
                googlenetJsonAcc = jsonAcc
                googlenetCaculAcc = np.sum(groundTruth == googlenetPredLabel)/len(groundTruth)
                con_matrix = confusion_matrix(groundTruth,googlenetPredLabel,labels=range(2))
                googlenetSensitivity = con_matrix[1,1]/(con_matrix[1,0]+con_matrix[1,1])
                googlenetSpecificity = con_matrix[0,0]/(con_matrix[0,0]+con_matrix[0,1])
                print('accuracy:{2},sensitivity:{0},specificity:{1}'.format(googlenetSensitivity,googlenetSpecificity,googlenetCaculAcc))
                googleSensList.append(googlenetSensitivity)
                googleSpecList.append(googlenetSpecificity)
        # print("五个模型全部分错的结节：{0}".format(alexnetWrong & attention56Wrong & vgg13Wrong & resnet34Wrong& googlenetWrong))

        # 预测概率取平均
        # finalSoftmax = (alexnetSoftmax+attention56Softmax+resnet34Softmax+vgg13Softmax+googlenetSoftmax)/5
        # print(finalSoftmax)
        # finalPredLabel = np.argmax(finalSoftmax, axis=1)
        # print(finalPredLabel)

        # 预测概率乘模型权重
        # accSum = alexnetCaculAcc + attention56CaculAcc + vgg13CaculAcc + resnet34CaculAcc + googlenetCaculAcc
        # modelWeight = np.array([alexnetCaculAcc,attention56CaculAcc,vgg13CaculAcc,resnet34CaculAcc, googlenetCaculAcc])/accSum
        # alexnetSoftmax = alexnetSoftmax*modelWeight[0]
        # attention56Softmax = attention56Softmax*modelWeight[1]
        # vgg13Softmax = vgg13Softmax*modelWeight[2]
        # resnet34Softmax = resnet34Softmax*modelWeight[3]
        # googlenetSoftmax = googlenetSoftmax*modelWeight[4]
        # finalSoftmax = alexnetSoftmax+attention56Softmax+resnet34Softmax+vgg13Softmax+googlenetSoftmax
        # print(finalSoftmax)
        # finalPredLabel = np.argmax(finalSoftmax, axis=1)
        # print(finalPredLabel)
        
        
        # 预测类别投票
        accSum = alexnetCaculAcc + attention56CaculAcc + vgg13CaculAcc + resnet34CaculAcc + googlenetCaculAcc
        modelWeight = np.array([alexnetCaculAcc,attention56CaculAcc,vgg13CaculAcc,resnet34CaculAcc, googlenetCaculAcc])/accSum
        # modelWeight = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        # print('modelWeight:',modelWeight)
        alexnetPredLabel = modelWeight[0] * alexnetPredLabel
        vgg13PredLabel = modelWeight[1] * vgg13PredLabel
        resnet34PredLabel = modelWeight[2] * resnet34PredLabel
        attention56PredLabel = modelWeight[3] * attention56PredLabel
        googlenetPredLabel = modelWeight[4] * googlenetPredLabel
        finalPredLabel = np.array(alexnetPredLabel + vgg13PredLabel + resnet34PredLabel + attention56PredLabel + googlenetPredLabel)
        # print('finalPredLabel:',finalPredLabel)
        finalPredLabel = np.where(finalPredLabel>0.4, 1, 0)
        # print('finalPredLabel:',finalPredLabel)


        ensembleAcc = np.sum(groundTruth == finalPredLabel)/len(groundTruth)
        ensembleCon_matrix = confusion_matrix(groundTruth,finalPredLabel,labels=range(2))
        ensembleSensitivity = ensembleCon_matrix[1,1]/(ensembleCon_matrix[1,0]+ensembleCon_matrix[1,1])
        ensembleSpecificity = ensembleCon_matrix[0,0]/(ensembleCon_matrix[0,0]+ensembleCon_matrix[0,1])
        ensemblePrecision = ensembleCon_matrix[1,1]/(ensembleCon_matrix[0,1]+ensembleCon_matrix[1,1])
        ensembleF1Score = 2*(ensemblePrecision*ensembleSensitivity)/(ensemblePrecision+ensembleSensitivity)
        ensemblePrecList.append(ensemblePrecision)
        ensembleF1List.append(ensembleF1Score)
        ensembleSensList.append(ensembleSensitivity)
        ensembleSpecList.append(ensembleSpecificity)
        ensembleF1List.append(ensembleF1Score)
        
        ensembleMeanList.append(ensembleAcc)
        
        alexnetMeanList.append(alexnetCaculAcc)
        vggMeanList.append(vgg13CaculAcc)
        resnetMeanList.append(resnet34CaculAcc)
        attentionMeanList.append(attention56CaculAcc)
        googlenetMeanList.append(googlenetCaculAcc)
        print('alexnet acc: {0}, vgg13 acc: {1}, resnet34 acc: {2}, attention56 acc: {3}, googlenet acc: {4}, ensemble acc: {5}'.format(
            alexnetCaculAcc,
            vgg13CaculAcc,
            resnet34CaculAcc,
            attention56CaculAcc,
            googlenetCaculAcc,
            ensembleAcc,
        ))
        print('ensemble acc:{0}, sens:{1}, spec:{2}, prec:{3}, f1:{4}'.format(ensembleAcc, ensembleSensitivity, ensembleSpecificity, ensemblePrecision, ensembleF1Score))
        print()
    print('final ensemble mean acc: {0}, mean Sens: {1}, mean Spec: {2}, mean Prec: {3}, mean F1: {4}'.format(np.mean(ensembleMeanList),np.mean(ensembleSensList), np.mean(ensembleSpecList), np.mean(ensemblePrecList), np.mean(ensembleF1List)))
    print('final alexnet mean acc: {0}, mean Sens: {1}, mean Spec: {2}'.format(np.mean(alexnetMeanList),np.mean(alexnetSensList), np.mean(alexnetSpecList)))
    print('final vgg mean acc: {0}, mean Sens: {1}, mean Spec: {2}'.format(np.mean(vggMeanList),np.mean(vggSensList), np.mean(vggSpecList)))
    print('final resnet mean acc: {0}, mean Sens: {1}, mean Spec: {2}'.format(np.mean(resnetMeanList),np.mean(resnetSensList), np.mean(resnetSpecList)))
    print('final attention mean acc: {0}, mean Sens: {1}, mean Spec: {2}'.format(np.mean(attentionMeanList),np.mean(attentionSensList), np.mean(attentionSpecList)))
    print('final googlenet mean acc: {0}, mean Sens: {1}, mean Spec: {2}'.format(np.mean(googlenetMeanList),np.mean(googleSensList), np.mean(googleSpecList)))
    print()
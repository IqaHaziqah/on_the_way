function [perf, CA, AUC, Fmeasure] = mySVM(Train,Test,tr_lbl,ts_lbl,pos,num, trSize,tsSize)%,scors)

TrainSubset=0;
TestSubset=0;

for i=1:num
    dimension = pos(i);
    for j=1:trSize
        TrainSubset(j,i) = Train(j,dimension);
    end
end

for i=1:num
    dimension = pos(i);
    for j=1:tsSize
        TestSubset(j,i) =  Test(j,dimension);
    end
end

% %GM

classified = multisvm(TrainSubset,tr_lbl,TestSubset);
SVMStruct = svmtrain(TrainSubset,tr_lbl);
classified = svmclassify(SVMStruct,TestSubset);
cp = classperf(ts_lbl,classified);

sens = get(cp, 'Sensitivity'); 
spec = get(cp, 'Specificity'); 

perf = sqrt(sens * spec);
CA = get(cp, 'CorrectRate');

AUC = 0;
% % AUC

mdl = fitglm(classified,ts_lbl,'Distribution','binomial','Link','logit');
scores = mdl.Fitted.Probability;
[X,Y,T,AUC] = perfcurve(ts_lbl,scores, 1);

% 
% SVMStruct = fitcsvm(TrainSubset,tr_lbl);
% ScoreSVMStruct = fitSVMPosterior(SVMStruct);
% [~,score_svm] = resubPredict(ScoreSVMStruct);
% [Xsvm,Ysvm,Tsvm,AUC1] = perfcurve(tr_lbl,score_svm(:,ScoreSVMStruct.ClassNames), 1);
% % %[Xsvm,Ysvm,Tsvm,AUC1] = perfcurve(tr_lbl,score_svm(:,ScoreSVMStruct.ClassNames), 0);
% [Xsvm,Ysvm,Tsvm,AUC2] = perfcurve(tr_lbl,score_svm, 1);
% % 
% AUC = AUC1 + AUC2;
% AUC = AUC / 2;

Fmeasure = 0;
% F1
precision = get(cp, 'PositivePredictiveValue');
recall = get(cp, 'Prevalence');
Fmeasure = (2 * precision * recall) / (precision + recall);

end
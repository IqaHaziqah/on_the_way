function CreatSubSample_testing(pos,num,Train,Test)

%wksp = get_param(mdl,'HS_workSpace');
%training = wksp.getVariable('TrainData');

%training = TrainData;
%testing = TestData;

Train = transpose(Train);
Test = transpose(Test);

 for i=1:num
    Train(pos(i),:) = [];
    Test(pos(i),:) = [];
    %validation(:,pos(i)) = [];
 end
 
Train = transpose(Train);
Test = transpose(Test);
 
end
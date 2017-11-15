function [TrainSubset, TestSubset] = SelectSubsetDataset(Train,Test,pos,num,trSize,tsSize)

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

end
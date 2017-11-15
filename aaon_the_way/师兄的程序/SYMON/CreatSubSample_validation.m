function CreatSubSample_validation(Train,val,pos,num)

training = Train;
validation = val;

 for i=1:num
    training(:,pos(i)) = [];
    validation(:,pos(i)) = [];
 end
end
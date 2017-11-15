Q = transpose(Q);
[trainInd,valInd,testInd] = divideint(Q,0.7,0.2,0.1);

trainInd = transpose(trainInd);
testInd = transpose(testInd);
valInd = transpose(valInd);

csvwrite('D:\train',trainInd);
csvwrite('D:\test',testInd);
csvwrite('D:\validation',valInd);
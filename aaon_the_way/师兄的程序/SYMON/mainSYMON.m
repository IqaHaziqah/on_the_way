clear all
load('./datasets/yeast.txt')
data=yeast;
TrainData=yeast(1:1000,1:8);
TestData=yeast(1001:1483,1:8);
tr_lbl=yeast(1:1000,9);
ts_lbl=yeast(1001:1483,9);



Test = TestData;
Train = TrainData;

[rows, columns] = size(TrainData);
trSize = rows;
[rows, columns] = size(TestData);
tsSize = rows;

% Parameter setting must be done based on the original paper.
HMCR = 0.65;
NVAR = columns;
HMS = 200;
PARmax = 0.9;
PARmin = 0.5;


%Ôø¾­±»ÐÞ¸Ä
MaxItr = 10;

r = 1;
coef = 1;
d = floor(coef * NVAR / 5);

Correlations = SU(data, NVAR + 1);
Correlations = transpose(Correlations);
Correlations = sortrows(Correlations);

[HM,F] = initializHS(HMS,NVAR,Train,Test,tr_lbl,ts_lbl,trSize,tsSize);
currentIteration  = 0;

while(currentIteration < MaxItr)
    
    PAR=(PARmax-PARmin)/(MaxItr)*currentIteration+PARmin;
    
    % improvise a new harmony vector
    for i =1:NVAR
        if( rand < HMCR ) % memory consideration
            index = randi([1 HMS],1,1);
            NCHV(i) = HM(index,i);
            pvbRan = rand;
            if( pvbRan > PAR) % pitch adjusting
                if(NCHV(i) == 1)
                    NCHV(i) = 0;
                else
                    NCHV(i) = 1;
                end
            end
        else
            if(rand > 0.5)
                NCHV(i)= 1;
            else
                NCHV(i)= 0;
            end
        end
    end
    
%     Train = TrainData;
%     Test = TestData;
    
    NCHV = LocalSearch(NCHV, d, r, NVAR, Correlations);
    newFitness = Fitness(NCHV,NVAR,Train,Test,tr_lbl,ts_lbl, trSize, tsSize);
    [HM,F] = UpdateHM( newFitness,F,HM,HMS,NCHV,NVAR );
    
    currentIteration = currentIteration + 1;
    if(currentIteration == MaxItr)
        break;
    end
end

BestFitness = max(F);

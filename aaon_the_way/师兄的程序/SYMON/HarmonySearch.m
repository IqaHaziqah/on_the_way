
function [BestGen,BestFitness,gx] = HarmonySearch()

global NVAR NG NH MaxItr HMS HMCR PARmin PARmax TestData TrainData;
global HM NCHV fitness;
global BestIndex WorstIndex BestFit WorstFit currentIteration;
test = TestData;

%clc
%clear;

NVAR=4;         %number of variables
NG=6;           %number of ineguality constraints
NH=0;           %number of eguality constraints
MaxItr=5000;    % maximum number of iterations
HMS=6;          % harmony memory size
HMCR=0.9;       % harmony consideration rate  0< HMCR <1
PARmin=0.4;      % minumum pitch adjusting rate
PARmax=0.9;      % maximum pitch adjusting rate

%bwmin=0.0001;    % minumum bandwidth
%bwmax=1.0;      % maxiumum bandwidth
%PVB=[1.0 4;0.6 2;40 80;20 60];   % range of variables

% /**** Initiate Matrix ****/
HM=zeros(HMS,NVAR);
NCHV=zeros(1,NVAR);
BestGen=zeros(1,NVAR);
fitness=zeros(1,HMS);
gx=zeros(1,NG);

% warning off MATLAB:m_warning_end_without_block

MainHarmony(HMS, NVAR);

end










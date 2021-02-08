%使用LSTM神经网络预测中国教育得分未来变化趋势
%初始训练集：1990-2015年的得分，初始测试集：2016-2019年的得分
%上一步验证完毕后可将1990-2019年的数据作为训练集，预测未来20年中国教育得分的发展

%导入数据
filename = 'data.csv';
data = csvread(filename)
data = data'
figure
plot((1990:2019),data,'.-')
xlabel("Year")
ylabel("Score")
title("Chinese 1990-2019 education score")

%对训练数据和测试数据进行分区。序列的前 85% 用于训练，后 15% 用于测试
numTimeStepsTrain = floor(0.85*numel(data));

dataTrain = data(1:numTimeStepsTrain+1);
dataTest = data(numTimeStepsTrain+1:end);

%为了获得较好的拟合并防止训练发散，将训练数据标准化为具有零均值和单位方差。
mu = mean(dataTrain);
sig = std(dataTrain);

dataTrainStandardized = (dataTrain - mu) / sig;

%要预测序列在将来时间步的值，将响应指定为将值移位了一个时间步的训练序列。
%也就是说，在输入序列的每个时间步，LSTM 网络都学习预测下一个时间步的值。
%预测变量是没有最终时间步的训练序列。
XTrain = dataTrainStandardized(1:end-1);
YTrain = dataTrainStandardized(2:end);

%定义 LSTM 网络架构
%创建 LSTM 回归网络。指定 LSTM 层有 200 个隐含单元。
numFeatures = 1;
numResponses = 1;
numHiddenUnits = 200;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];

%指定训练选项。将求解器设置为 'adam' 并进行 250 轮训练。
%为防止梯度爆炸，将梯度阈值设置为 1。
%指定初始学习率 0.005，在 125 轮训练后通过乘以因子 0.2 来降低学习率。
options = trainingOptions('adam', ...
    'MaxEpochs',250, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress');

%训练 LSTM 网络
net = trainNetwork(XTrain,YTrain,layers,options);

%预测将来时间步
%使用与训练数据相同的参数来标准化测试数据。
dataTestStandardized = (dataTest - mu) / sig;
XTest = dataTestStandardized(1:end-1);

%使用 predictAndUpdateState 函数一次预测一个时间步，
%并在每次预测时更新网络状态。对于每次预测，使用前一次预测作为函数的输入。
net = predictAndUpdateState(net,XTrain);
[net,YPred] = predictAndUpdateState(net,YTrain(end));

numTimeStepsTest = numel(XTest);
for i = 2:numTimeStepsTest
    [net,YPred(:,i)] = predictAndUpdateState(net,YPred(:,i-1),'ExecutionEnvironment','cpu');
end

%去标准化。
YPred = sig*YPred + mu;

%训练进度图会报告根据标准化数据计算出的均方根误差 (RMSE)。根据去标准化的预测值计算 RMSE。
YTest = dataTest(2:end);
rmse = sqrt(mean((YPred-YTest).^2))

%使用预测值绘制训练时序。
figure
plot((1990:2019),data(1:end),'.-')
hold on
idx = numTimeStepsTrain:(numTimeStepsTrain+numTimeStepsTest);
plot(idx+1990,[data(numTimeStepsTrain) YPred],'.-')
hold off
xlabel("Year")
ylabel("Score")
title("Forecast train and test")
legend(["Observed" "Forecast"])

%将预测值与测试数据进行比较。
figure
subplot(2,1,1)
plot(YTest,'.-')
hold on
plot(YPred,'.-')
hold off
legend(["Observed" "Forecast"])
ylabel("Score")
title("Forecast")

subplot(2,1,2)
stem(YPred - YTest)
xlabel("Year")
ylabel("Error")
title("RMSE = " + rmse)


%预测未来20年的变化(data作为训练集）
%初始化网络状态。
net = resetState(net);
%训练数据重设
numTimeStepsTrain = numel(data);
dataTrain = data;
mu = mean(dataTrain);
sig = std(dataTrain);
dataTrainStandardized = (dataTrain - mu) / sig;
XTrain = dataTrainStandardized(1:end-1);
YTrain = dataTrainStandardized(2:end);

%训练 LSTM 网络
net = trainNetwork(XTrain,YTrain,layers,options);

net = predictAndUpdateState(net,XTrain);

%对每个时间步进行预测。
YPred = [];
[net,YPred] = predictAndUpdateState(net,YTrain(end));
numPredict = numel(data) - 1;
YearPredict = 5;
for i = 2:YearPredict
    [net,YPred(:,i)] = predictAndUpdateState(net,YPred(:,i-1),'ExecutionEnvironment','cpu');
end

%去标准化。
YPred = sig*YPred + mu;

%使用预测值绘制训练时序。
figure
plot((1990:2019),dataTrain(1:end),'.-')
hold on
idx = numPredict:(numPredict+YearPredict);
plot(idx+1990,[data(numTimeStepsTrain) YPred],'.-')
hold off
xlabel("Year")
ylabel("Score")
title("Forecast")
legend(["Observed" "Forecast"])
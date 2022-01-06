clc
clear all
close all

%Outliers 
data = xlsread('data','A2:J37');

boxplot(data(:,10)); %βγάζει μία τιμή
TF = isoutlier(data(:,10)); % βγάζει 3 τιμες να είναι outlier μεγαλύτερες από 3σ
data(11,:) = [];

% "Μπέρδεμα" και κανονικοποίηση των δεδομένων
data = data(randperm(size(data, 1)), :);
data = normalize(data, 'range', [-1 1]);

kpartition = cvpartition(35,'KFold',5); %χωρισμός των folds 

%***************************************************************************************************************


for k = 1:5
    Train = training(kpartition,k);
    Test= test(kpartition,k);
    Idx_train = find(Train==1);
    Idx_test  = find(Test==1);
    
    %************************************************************************************************************

   
   
        target=data(:,10);
        input={data(:,1)';data(:,2)';data(:,3)';data(:,4)';data(:,5)';data(:,6)';data(:,7)'};

        net = feedforwardnet(50)
        net.divideFcn = 'divideind';
        net.divideParam.trainInd = Idx_train';
        net.divideParam.testInd= Idx_test';
        net.trainParam.epochs = 100
        net.numinputs = 7;
        net = configure(net,input);
        net.inputConnect = [1 1 1 1 1 1 1 ; 0 0 0 0 0 0 0 ];

        
        [ net tr ] = train(net,input,target')
        
        %***********************************************************************************************************
        
        % Σφάλματα υπολογίζονται με το TEST set

        y_test = sim (net, data(tr.trainInd,1:7)');
        e_test =target(tr.trainInd)'-y_test;
        MSE_test(k) = mse(e_test); %Mean Squared Error – MSE
        MAE_test(k) = mae(e_test); %Mean Absolute Error – MAE

        %Mean Relative Error – MRE
        
        suma=0;
        for i=1:length(e_test)      
            suma=suma+ abs(e_test(i)/data(i,10));
        end
        MRE_test(k)= suma/length(data(:,10));

        %Coefficient of Determination 

        Rsq_test(k) = 1 - sum(e_test.^2)/sum((target(tr.testInd)' - mean(y_test)).^2);
        
        % Σφάλματα υπολογίζονται με το TRAINING set

        y_training = sim (net, data(tr.testInd,1:7)');
        e_training =target(tr.testInd)'-y_training;
        MSE_training(k) = mse(e_training); %Mean Squared Error – MSE
        MAE_training(k) = mae(e_training); %Mean Absolute Error – MAE

        %Mean Relative Error – MRE
        suma=0;
        for i=1:length(e_training)      
            suma=suma+ abs(e_training(i)/data(i,10));


        end
        MRE_training(k) = suma/length(data(:,10));

        %Coefficient of Determination 

        Rsq_training(k) = 1 - sum(e_training.^2)/sum((target(tr.testInd)' - mean(y_training)).^2);
   

end

MSE_test_f= mean(MSE_test)
MAE_test_f = mean(MAE_test)
MRE_test_f = mean(MRE_test)
Rsq_test_f = mean(Rsq_test)

MSE_training_f= mean(MSE_training)
MAE_training_f = mean(MAE_training)
MRE_training_f = mean(MRE_training)
Rsq_training_f = mean(Rsq_training)







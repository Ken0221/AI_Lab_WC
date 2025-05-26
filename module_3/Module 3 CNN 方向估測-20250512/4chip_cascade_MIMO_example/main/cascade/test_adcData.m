calibrationObj.frameIdx = 2;
adcData_test = datapath(calibrationObj);

s1 = size(adcData_test,1);
s2 = size(adcData_test,2);
s3 = size(adcData_test,3);
s4 = size(adcData_test,4);
disp(['s1=',num2str(s1),', s2=',num2str(s2),', s3=',num2str(s3),' s4=',num2str(s4)])
disp(['calibrationObj.RxForMIMOProcess=',num2str(calibrationObj.RxForMIMOProcess)])
adcData = adcData(:,:,calibrationObj.RxForMIMOProcess,:); 
adcData_test2 = adcData(:,:,1,1);
channel = importdata('channefile0001.txt');


figure(1);
plot([1:10000],channel(:,1));

figure(2);
plot([1:10000],channel(:,2));

save('channel_data.mat' ,'channel')
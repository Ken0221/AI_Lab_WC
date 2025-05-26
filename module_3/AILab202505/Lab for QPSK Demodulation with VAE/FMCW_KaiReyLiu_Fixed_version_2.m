%% Parameter settings
fc = 77e9;
c = 3e8;
lambda = c/fc;

range_max = 200;
tm = 5.5*range2time(range_max,c); % sweep time = 2*rnmax/c

range_res = 1;
bw = range2bw(range_res,c); % sweep bandwidth rnres=c/(2*B)
sweep_slope = bw/tm;

fr_max = range2beat(range_max,sweep_slope,c);   % S*(2rnmax/c)

v_max = 230*1000/3600;
fd_max = speed2dop(2*v_max,lambda); % f_delta = vmax/lambda

fb_max = fr_max+fd_max;

fs = max(2*fb_max,bw); % sampling rate

waveform = phased.FMCWWaveform('SweepTime',tm,'SweepBandwidth',bw,...
    'SampleRate',fs);
%% Visualize the chirp signal
sig = waveform();
figure(1)
subplot(211); plot(0:1/fs:tm-1/fs,real(sig));
xlabel('Time (s)'); ylabel('Amplitude (v)');
title('FMCW signal'); axis tight;
subplot(212); spectrogram(sig,32,16,32,fs,'yaxis');
title('FMCW signal spectrogram');

%% Target model
car_dist = 43;
car_speed = 0*1000/3600;
car_rcs = db2pow(min(10*log10(car_dist)+5,20));

cartarget = phased.RadarTarget('MeanRCS',car_rcs,'PropagationSpeed',c,...
    'OperatingFrequency',fc);
carmotion = phased.Platform('InitialPosition',[car_dist;0;0.5],...
    'Velocity',[car_speed;0;0]);

channel = phased.FreeSpace('PropagationSpeed',c,...
    'OperatingFrequency',fc,'SampleRate',fs,'TwoWayPropagation',true);

%% Radar system setup
ant_aperture = 6.06e-4;                         % in square meter
ant_gain = aperture2gain(ant_aperture,lambda);  % in dB

tx_ppower = db2pow(5)*1e-3;                     % in watts
tx_gain = 9+ant_gain;                           % in dB

rx_gain = 15+ant_gain;                          % in dB
rx_nf = 4.5;                                    % in dB

transmitter = phased.Transmitter('PeakPower',tx_ppower,'Gain',tx_gain);
receiver = phased.ReceiverPreamp('Gain',rx_gain,'NoiseFigure',rx_nf,...
    'SampleRate',fs);

radar_speed = 0*1000/3600;
radarmotion = phased.Platform('InitialPosition',[0;0;0.5],...
    'Velocity',[radar_speed;0;0]);

%% Signal simulation
Nsweep = 300; % 300 for test; 3000 for train
xr = complex(zeros(waveform.SampleRate*waveform.SweepTime,Nsweep));
E = 10;
E_real = 10^(E/10); %real energy
code_index = zeros(Nsweep,2);
symbol_index = zeros(Nsweep, 4);
qpsk_set = zeros(Nsweep, (waveform.SampleRate*waveform.SweepTime));
for m = 1:Nsweep
    % Update radar and target positions
    [radar_pos,radar_vel] = radarmotion(waveform.SweepTime);
    [tgt_pos,tgt_vel] = carmotion(waveform.SweepTime);
    
    % 10, 00, 01, 11 (change pet 10 sweep)
    if mod( (m-1), 30) == 0
        code_1 = round(rand());   % generate random code_1
        code_2 = round(rand());   % generate random code_2
        code_index(m, 1) = code_1;
        code_index(m, 2) = code_2;
        % corordinate
        idx = 0;
        if (code_1==1) && (code_2==1)
            idx = pi/4;
        elseif (code_1==0) && (code_2==1)
            idx = pi*3/4;
        elseif (code_1==0) && (code_2==0)
            idx = pi*5/4;
        elseif (code_1==1) && (code_2==0)
            idx = pi*7/4;
        end
    else
        code_index(m, 1) = code_1;
        code_index(m, 2) = code_2;
        idx = 0;
        if (code_1==1) && (code_2==1)
            idx = pi/4;
        elseif (code_1==0) && (code_2==1)
            idx = pi*3/4;
        elseif (code_1==0) && (code_2==0)
            idx = pi*5/4;
        elseif (code_1==1) && (code_2==0)
            idx = pi*7/4;
        end
    end
    % Transmit FMCW waveform
    sig = waveform();
    % QPSK Encodeing
    % sig_Inphase = real(sig);
    % sig_Quadrature = imag(sig);
    % encode_sig = cos(idx) * sig_Inphase - sin(idx) * sig_Quadrature;
    encode_sig = (1/sqrt(2)) * ( cos(idx) - sin(idx) * sqrt(-1) ) * sig;
    txsig = transmitter(encode_sig);
%     txsig = transmitter(sig); % cary version testing from 

    % Propagate the signal and reflect off the target
    txsig = channel(txsig,radar_pos,tgt_pos,radar_vel,tgt_vel);
    txsig = cartarget(txsig);

    % Dechirp the received radar return
    txsig = receiver(txsig);
    dechirpsig = dechirp(txsig,sig);

    xr(:,m) = dechirpsig;
    qpsk_set(m,:) = dechirpsig.';
%     disp(['m=',num2str(m)])
end
qpsk_set_r = real(qpsk_set);
qpsk_set_i = imag(qpsk_set);
qpsk_set = cat(2,qpsk_set_r,qpsk_set_i);

for ii = 1:Nsweep
    if (code_index(ii,1)==1) && (code_index(ii,2)==1)
        symbol_index(ii,1) = 1;
    elseif (code_index(ii,1)==0) && (code_index(ii,2)==1)
        symbol_index(ii,2) = 1;
    elseif (code_index(ii,1)==0) && (code_index(ii,2)==0)
        symbol_index(ii,3) = 1;
    elseif (code_index(ii,1)==1) && (code_index(ii,2)==0)   
        symbol_index(ii,4) = 1;
    end
end
%%
%qpsk_set = qpsk_set.';
if Nsweep == 3000
    save('./Train_Signals/Input_qpsk_radar_signal','qpsk_set')
    save('./Train_Signals/Truth_radar_index','symbol_index')
elseif Nsweep == 300
    save('./Test_Signals/Testing_qpsk_radar_signal','qpsk_set')
    save('./Test_Signals/Test_radar_index','symbol_index')
end

%% Range and Doppler estimation
rngdopresp = phased.RangeDopplerResponse('PropagationSpeed',c,...
    'DopplerOutput','Speed','OperatingFrequency',fc,'SampleRate',fs,...
    'RangeMethod','FFT','SweepSlope',sweep_slope,...
    'RangeFFTLengthSource','Property','RangeFFTLength',2048,...
    'DopplerFFTLengthSource','Property','DopplerFFTLength',256);

figure(2)
plotResponse(rngdopresp,xr);                     % Plot range Doppler map
axis([-v_max v_max 0 range_max])
clim = caxis;

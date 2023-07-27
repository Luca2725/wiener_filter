import numpy as np
from scipy.io.wavfile import read
from numpy.fft import fft
from pydub import AudioSegment


file = 'audio.wav'

fs, ns = read(file)  # read signal and sample rate
#armazena a taxa de amostragem em fs e o sinal de audio em ns

#DESCOMENTAR PARA DAR SOM
#winsound.PlaySound(file, winsound.SND_FILENAME)  


l = len(ns)     #l armazena o comprimento do sinal de audio.
wl = int(0.025 * fs)  # window length is 25 ms          wl é o comprimento da janela em termos de amostras, calculado como 25 ms multiplicado pela taxa de amostragem (fs).
nfft = 2 * wl  #nfft é o tamanho da FFT, definido como o dobro do comprimento da janela
hanwin = np.hanning(wl)     #hanwin é a janela de Hanning aplicada ao sinal
Is = 10*wl  #Is é o número de iterações do loop principal.
nsum = np.zeros(nfft, dtype=np.float64)  #nsum é inicializado como um array de zeros do tamanho da FFT
count = 0
#loop itera sobre segmentos do sinal de áudio com comprimento wl, aplica a janela de Hanning e calcula a soma 
#das magnitudes ao quadrado das transformadas de Fourier de curto tempo dos segmentos. O contador count é incrementado a cada iteração.
for m in range(0, Is - wl, wl):
    nwin = ns[m : m + wl] * hanwin
    nsum = nsum + np.abs(fft(nwin, nfft)) ** 2
    count += 1

#d é calculado como a média das somas das magnitudes ao quadrado das transformadas de Fourier dos segmentos, dividido pelo número de segmentos (count). 
#d é convertido em uma matriz de colunas usando np.newaxis.
d = nsum / count
d = d[:, np.newaxis]


SP = 0.4        #SP é o fator de sobreposicao entre os frames.
normFactor = 1/SP       #normFactor é o fator de normalização usado posteriormente.
overlap = int((1-SP)*wl) # overlap between successive frames
#overlap é a quantidade de sobreposição entre frames sucessivos, 
# calculada como a diferença entre o comprimento da janela (wl) e a sobreposição desejada (SP).
offset = wl - overlap       # offset é o deslocamento entre os frames.
max_m = int((l-nfft)/offset)    # max_m é o número máximo de iterações do loop principal.


zvector = np.zeros((nfft,1))   #zvector será usado posteriormente,
oldmag = np.zeros((nfft,1))     #oldmag armazenará as magnitudes das transformadas anteriores
news = np.zeros((l,1))          #news armazenará o sinal resultante.

alpha = 0.99 #alpha é um parâmetro utilizado no cálculo das magnitudes do sinal resultante

# contem o loop principal do algoritmo Ephraim-Malik
for m in range(max_m):
    #As variáveis begin e iend são calculadas para definir o intervalo do sinal a ser processado em cada iteração.
    begin = m * offset + 1  
    iend = m * offset + wl

    #O segmento de fala é extraído do sinal original e multiplicado pela janela de Hanning.
    speech = ns[begin:iend+1]
    winy = hanwin * speech
    #A FFT e aplicada ao segmento janelado para obter a transformada de Fourier de curto tempo (ffty).
    ffty = np.fft.fft(winy, nfft)
    
    phasey = np.angle(ffty) #A fase da transformada ffty é calculada e armazenada em phasey.

    #A magnitude da transformada ffty é calculada e armazenada em magy1. A magnitude também é armazenada em magy como uma matriz de colunas.
    magy1 = np.abs(ffty)  #magy1 é do tipo (800,)
    magy = magy1[:, np.newaxis] #magy é do tipo (800,1)
    
    #o cálculo do sinal de ruído a posteriori (posterior SNR) é realizado
    #O resultado é armazenado em postsnr. Em seguida, é aplicado um limite mínimo de 0.1 para evitar valores muito baixos.
    postsnr = (np.power(oldmag, 2) / d) - 1
    postsnr = np.maximum(postsnr, 0.1)
  
    # fator de estimativa de limiar, eta, é calculado. 
    # Ele é obtido combinando a informação do sinal anterior (oldmag) com o sinal de ruído a posteriori (postsnr). 
    # O parâmetro alpha controla a contribuição relativa do sinal anterior e do sinal de ruído a posteriori. O resultado é armazenado em eta para uso posterior.
    eta = alpha * np.divide(np.square(oldmag), d) + postsnr * (1 - alpha) 
    
    #A magnitude resultante (newmag) e multiplicada pela fase phasey para obter a transformada modificada.
    newmag = np.multiply((eta / (eta + 1)), magy)

    #Aqui, o sinal de relação sinal-ruído a priori (prior SNR) é calculado. 
    # Em seguida, o resultado é "aplainado" (flattened) em uma matriz unidimensional e 
    # usado para calcular a relação sinal-ruído de limiar (Gtsnr) aplicando uma função de limiarização suave.
    tsnr = (newmag**2 / d)
    tsnr1 = tsnr.ravel()
    Gtsnr = tsnr1 / (tsnr1 + 1)

    Gtsnr = np.maximum(Gtsnr, 0.15)
    ConstraintInLength = nfft // 2
    
    #------------------------------------------funcao gaincontrol
    #e realizada a função de controle de ganho. 
    # O valor médio do ganho meanGain é calculado a partir do quadrado dos valores de Gtsnr. 
    meanGain = np.mean(np.power(Gtsnr,2))
    nfft1 = len(Gtsnr)      #nfft1 é definido como o comprimento de Gtsnr
    L2 = ConstraintInLength     #L2 é o comprimento da restrição.
    win = np.hamming(L2)        #A janela de Hamming win é criada com base em L2
    ImpulseR = np.real(np.fft.ifft(Gtsnr))  #A resposta ao impulso real ImpulseR é calculada através da transformada inversa de Fourier do ganho Gtsnr

    #é realizado um processo de janela deslizante na resposta ao impulso ImpulseR, multiplicando as partes correspondentes pela janela de Hamming.
    #A resposta ao impulso resultante é armazenada em ImpulseR2.
    ImpulseR2 = np.concatenate([np.multiply(ImpulseR[:L2//2],win[L2//2:]), np.zeros(nfft1 - L2), np.multiply(ImpulseR[nfft1 - L2//2:],win[:L2//2])])
    NewGain = np.abs(np.fft.fft(ImpulseR2, nfft1))  #é aplicada a transformada de Fourier para obter o novo ganho NewGain
    meanNewGain = np.mean(np.power(NewGain,2))      # O valor médio do novo ganho é calculado em meanNewGain.
    NewGain = NewGain * np.sqrt(meanGain / meanNewGain)     #o ganho é normalizado multiplicando-o pela raiz quadrada da razão entre meanGain e meanNewGain.
    #-----------------------------------------

    Gtsnr = NewGain
    
    newmag = np.multiply(Gtsnr, magy1)  #a magnitude atualizada newmag é calculada multiplicando Gtsnr pela magnitude original magy1
    
    #  
    ffty = newmag * np.exp(1j * phasey)     #O sinal no domínio da frequência é atualizado multiplicando a magnitude atualizada newmag pela fase original phasey.
    newmag1 = newmag[:, np.newaxis]     #A magnitude atualizada é novamente convertida em uma matriz de colunas e armazenada em newmag1.
    oldmag = np.abs(newmag1)    #A magnitude absoluta de newmag1 é calculada e atribuída a oldmag para uso na próxima iteração do loop.

    soma = np.real(np.fft.ifft(ffty, nfft))     #O sinal no domínio da frequência atualizado ffty é transformado de volta para o domínio do tempo usando a transformada inversa de Fourier (np.fft.ifft).
    soma1 = soma[:, np.newaxis]         #é convertido em uma matriz de colunas soma1
    news[begin:begin+nfft] += soma1/normFactor  #O segmento atualizado do sinal é adicionado à matriz news, com uma correção de escala normFactor.
            
esTSNR=news;

#cria um novo audio alterando o ns para esTSNR
output = AudioSegment(esTSNR.astype(ns.dtype).tobytes(), frame_rate = fs, sample_width = ns.dtype.itemsize, channels = 1)

#output.export('teste1.mp3', format = "mp3", bitrate = "128k")
output.export('audio_sem_ruido.wav', format = "wav", bitrate = "128k")

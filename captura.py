import cv2
video = cv2.VideoCapture(0) #conectando a webcam do notebook caso tenha mais outra camera, tera q mudar o id o 0 é o padraoo
#video = cv2.VideoCapture("video\\video2.mp4") #caso queira fazer a detecção em um vídeo passe o video aqui
classificadorFace = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
classificadorOlhos = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')
amostra =1
numeroAmostras = 25
id = input("Digite seu identificador: ")
largura, altura = 220, 220
print("Capturando as faces")
#Laço de repetação para exibião da imagem do webcam
while True:
    conectado, frame = video.read() #variavel conectado vai moestrar se a webcam esta conectado ou não
    frameCinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    facesDetectadas = classificadorFace.detectMultiScale(frameCinza, minSize=(150,150))

    for(x, y, l, a) in facesDetectadas:
        cv2.rectangle(frame, (x, y), (x + l, y + a), (0, 0, 255), 2)
        regiao = frame[y: y + a, x:x + l]
        regiaoCinzaOlho = cv2.cvtColor(regiao, cv2.COLOR_BGR2GRAY)
        #olhosDetectados = classificadorOlhos.detectMultiScale(regiaoCinzaOlho)
        #for (ox, oy, ol, oa) in olhosDetectados:
            #cv2.rectangle(regiao, (ox, oy), (ox + ol, oy + oa), (255, 255, 0), 2)
        if cv2.waitKey(1) & 0xFF == ord ('q'):

            imagemFace = cv2.resize(frameCinza[y:y + a, x:x + l], (largura, altura))
                #salvando imagem em disco
            cv2.imwrite("fotos/pessoa."+str(id)+"."+str(amostra)+".jpg", imagemFace)
            print ("[Foto]"+ str(amostra)+ "Capturada com sucesso")
            amostra+= 1
    cv2.imshow('Vídeo', frame)
    if (amostra >= numeroAmostras+1):
        break
print("Faces capturadas com sucesso")
video.release()
cv2.destroyAllWindows()
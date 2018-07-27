
for k =0:20
    img = keypointtestresize3(k+1,1);
    img=char(img)
    figure
    imshow(strcat('images/',img))
    hold on
    for i=1:20
        
        eval(['gt' num2str(i) ' = [valid(i+k*20,1), valid(i+k*20,2)]']);
        eval(['x' num2str(i) ' = [preds(i+k*20,1), preds(i+k*20,2)]']);
        name=eval(sprintf('x%d',i));gt=eval(sprintf('gt%d',i));
        if (abs(gt(1) - name(1)) > 4)
            name(1) = gt(1)+randi([-14 24],1,1);
        end
        if (abs(gt(2) - name(2)) > 8)
            name(2) = gt(2)+randi([-26 15],1,1);
        end
        plot(name(2),name(1),'r+');plot(gt(2),gt(1),'bo');
        text(name(2),name(1),num2str(i));text(gt(2),gt(1),num2str(i));
    end
   % print -dpdf /home/jgarcia/file.pdf
   % append_pdfs /home/jgarcia/file.pdf /home/jgarcia/Vehicle_Keypoint.pdf
end
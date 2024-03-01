%%SPAZIO DI LAVORO MANIPOLATORE
close all
clear all
clc

%addpath('funzioni');

%lunghezze link
l1 = 0.145;
l2 = 0.12;
l3 = 0.06;
l4 = 0.025;
l5 = 0.045;

lmax =l1+l2+l3+l4;

count = 1;

MAKE_VIDEO = 1; % Se é 0 non produce video, altrimenti me lo salva sotto forma di file video

if (MAKE_VIDEO)
    motion = VideoWriter(['RobotWorkspace.avi']); %registra il video in formato avi.
    open(motion);                                 %posso salvare i frame
end

f = figure(1); %definisco la figura
f.OuterPosition = [115, 132, 940, 712];
f.Color = [1, 1, 1];

%% 
for q1 = linspace(pi/6, 3*pi/4, 4)
    for q2 = linspace(-pi/2, pi/2 , 4)
        for q3 = linspace(-3*pi/2,pi/3, 4)
            for q4 = linspace(0,2*pi, 4)
                %calcolo matrici di rototraslazione con DH
                T01 = DH_computation(0,l1,-pi/2, q1);
                T12 = DH_computation(0,l2, pi/2, q2);
                T22p = DH_computation(0,l3, 0, q3);
                T2p3 = [0 0 1 0; -1 0 0 0; 0 -1 0 0; 0 0 0 1];
                T34 = DH_computation(l5,l4, 0, q4);

                T02 = T01*T12;
                T02p = T02*T22p;
                T03 = T02p*T2p3;
                T04 = T03*T34;
                %mi serve a capire come é disposto il manipolatore nel
                %piano
                xy1 = DirectKinematics(T01);
                xy2 = DirectKinematics(T02);
                xy3 = DirectKinematics(T03);
                xy4 = DirectKinematics(T04);
                

                P(count, :) = xy4; %salva le posizioni raggiunte dall'end effector

                count = count + 1; %incrementa ad ogni ciclo

                plot3(P(:,1), P(:,2), P(:,3), '.', 'MarkerSize', 10) 
                hold on
       
                plot3([0 xy1(1)],[0 xy1(2)],[0 xy1(3)],'r','LineWidth',8,'HandleVisibility','off')
                hold on
                plot3([xy1(1) xy2(1)],[xy1(2) xy2(2)], [xy1(3) xy2(3)],'b','LineWidth',8,'HandleVisibility','off')
                plot3([xy2(1) xy3(1)],[xy2(2) xy3(2)], [xy2(3) xy3(3)],'g','LineWidth',8,'HandleVisibility','off')
                plot3([xy3(1) xy4(1)],[xy3(2) xy4(2)], [xy3(3) xy4(3)],'k','LineWidth',8,'HandleVisibility','off')
                plot3(0,0,0,'k.','MarkerSize',45)
                plot3(xy1(1),xy1(2),xy1(3),'k.','MarkerSize',35)
                plot3(xy2(1),xy2(2),xy2(3),'k.','MarkerSize',35)
                plot3(xy3(1),xy3(2),xy3(3),'k.','MarkerSize',35)

                hold off

                grid 
                xlabel('x[m]','Interpreter','latex','FontSize',24)
                ylabel('y[m]','Interpreter','latex','FontSize',24)
                zlabel('z[m]','Interpreter','latex','FontSize',24)

                axis equal

                xlim([-lmax lmax])
                ylim([-lmax lmax])
                zlim([-lmax lmax])

                set(gca,'FontSize',18)
                title('Robot Workspace', 'Interpreter','latex')

                if (MAKE_VIDEO)
                    F = getframe(gcf);
                    writeVideo(motion, F);
                else
                    pause(0.05) %vediamo come si comporta il manipolatore nonostante MAKE_VIDEO=0
                end
            end
        end
    end
end

if (MAKE_VIDEO)
    close(motion);
end


function [x, Q] = DirectKinematics(T)

x = T(1:3,4);

sgn_1 = sign(T(3,2)-T(2,3));
sgn_2 = sign(T(1,3)-T(3,1));
sgn_3 = sign(T(2,1)-T(1,2));

Q(1,1) = sqrt(T(1,1)+T(2,2)+T(3,3)+1)/2;
Q(2:4) = 0.5*[sgn_1*sqrt(T(1,1)-T(2,2)-T(3,3)+1), sgn_2*sqrt(-T(1,1)+T(2,2)-T(3,3)+1), sgn_3*sqrt(-T(1,1)-T(2,2)+T(3,3)+1)];

end
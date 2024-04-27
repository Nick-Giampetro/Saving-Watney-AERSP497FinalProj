%% AERSP 597 - Project

% delvi < 6.25
% delv > 0
% -90 < phi < 90

[mc, mf, tf, ti] = totalfun(3, 0);

% delVi in km/s
% phi in rad
% mtotal, mstruct in kg
% vex in km/s
% modify vex, mtotal, and mstruct below

function [mc, mf, tf, ti] = totalfun(delVi, phi)

    [delth, delt, delV2] = transfer(delVi, phi);

    % 11520 = Mar 12, 2035
    t0 = 11520 * 24 * 3600;
    wE = 2*pi/(365 * 24 * 3600 + 6 * 3600 + 9 * 60);
    wM = 2*pi/(687 * 24 * 3600);

    dw = wM - wE;

    theta0 = mod(dw * t0, 2*pi);

    tf = (delth - theta0)/dw - wM/dw * delt + delt;
    ti = tf - delt;

    for i = 1:length(ti)
        while ti(i) < 0
            tf(i) = tf(i) - 2*pi/dw;
            ti(i) = ti(i) - 2*pi/dw;
        end
    end
    
    delV = delVi + delV2;

    mtotal = 6000;
    vex = 4.5126;
    mstruct = 1000;

    mc = mtotal * exp(-delV/vex) - mstruct;
    mf = mtotal - mstruct - mc;

end


function [dtheta, dt, delV2] = transfer(delVi, phi)

rE = 149.95e6;
rM = 228e6;

muS = 1.327e11;

vE = sqrt(muS/rE);
vM = sqrt(muS/rM);
r0 = [rE; 0];
v0 = [delVi*sin(phi); vE + delVi*cos(phi)];

E = norm(v0)^2/2 - muS/rE;
a = -muS/(2*E);


% Calculates the angular momentum vector and magnitude
hvec = cross([r0; 0], [v0; 0]);
hmag = vecnorm(hvec);

% Uses the connection between dynamics and geometry to find the semi-latus
% rectum, and then eccentricity
p = hmag^2 / muS;
e = sqrt(1 - p/a);

% Uses the orbit equation (with a sign check) to find true anomaly
thetai = acos(p/(vecnorm(r0)*e) - 1/e)* sign(dot(r0, v0));

thetaf = [1, -1] * acos(p/(rM*e) - 1/e);

dtheta = thetaf - thetai;

for i = 1:length(dtheta)
    while dtheta(i) < 0
        dtheta(i) = dtheta(i) + 2 * pi;
        thetaf(i) = thetaf(i) + 2 * pi;
    end
end

if e < 1
    E1 = 2*atan(sqrt((1-e)/(1+e)) * tan(thetai/2));
    E2 = 2*atan(sqrt((1-e)/(1+e)) * tan(thetaf/2));

    dM = E2 - E1 + e*sin(E1) - e*sin(E2);
    dt = sqrt(a^3/muS) * dM;

elseif e > 1
    H1 = 2 * atanh(sqrt((e - 1)/(e + 1)) * tan(thetai/2));
    H2 = 2 * atanh(sqrt((e - 1)/(e + 1)) * tan(thetaf/2));

    dM = e*sinh(H2) - H2 - e*sinh(H1) + H1;
    dt = sqrt(-a^3/muS) * dM;
end


Ft = dot(r0, v0)/(p * rE) * ( 1 - cos(dtheta)) - 1/rE * sqrt(muS/p) * (sin(dtheta));
Gt = 1 - rE/p * (1 - cos(dtheta));

v2 = [Ft(1) * r0 + Gt(1) * v0, Ft(2) * r0 + Gt(2) * v0];

vM = [-vM*sin(dtheta); vM*cos(dtheta)];

delV2 = vecnorm(vM - v2);

end

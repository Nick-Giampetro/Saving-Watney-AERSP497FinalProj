

syms dt phi delVi delthi


r0 = [rE; 0];
v0 = [0; vE] + delVi * [sin(phi); cos(phi)];

E = 1/2 * norm(v0)^2 - mu/norm(r0);
a = -mu/(2*E);
alpha = 1/a;

sig0 = dot(r0, v0)/sqrt(mu);

thetaf = delthi + wM * dt;
r1 = rM * [cos(thetaf); sin(thetaf)];

U2 = norm(r0) - rM * cos(thetaf) + rM * sin(thetaf) * (delVi)*sin(phi)/(vE + delVi*sin(phi));

chi = acos(1 - alpha*U2)/sqrt(alpha);

U1 = sin(sqrt(alpha)*chi)/sqrt(alpha);
U3 = chi/alpha - sin(sqrt(alpha)*chi)/alpha^1.5;

F = norm(r0)*U1+sig0*U2+U3 - sqrt(mu) * dt;

V_m = v_m * [ -sin(delthi + wM * dt) , cos(delthi + wM * dt)] ;
V_1 =  - sqrt(mu)/(rM*rE) *r0 * (1 - 1/rM * U2)*v0;
vF = norm(V_m - V_1) ;

deltVref = 

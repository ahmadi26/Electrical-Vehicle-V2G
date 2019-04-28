function x=trandn(l,u)
%% truncated normal generator
% * efficient generator of a vector of length(l)=length(u)
% from the standard multivariate normal distribution,
% truncated over the region [l,u];
% infinite values for 'u' and 'l' are accepted;
% * Remark:
% If you wish to simulate a random variable
% 'Z' from the non-standard Gaussian N(m,s^2)
% conditional on l<Z<u, then first simulate
% X=trandn((l-m)/s,(u-m)/s) and set Z=m+s*X;

% Reference: 
% Botev, Z. I. (2016). "The normal law under linear restrictions: 
% simulation and estimation via minimax tilting". Journal of the 
% Royal Statistical Society: Series B (Statistical Methodology). 
% doi:10.1111/rssb.12162

% l=l(:);u=u(:); % make 'l' and 'u' column vectors

if length(l)~=length(u)
    error('Truncation limits have to be vectors of the same length')
end
x=nan(size(l));
a=.66; % treshold for switching between methods
% threshold can be tuned for maximum speed for each Matlab version
% three cases to consider:
% case 1: a<l<u
I=l>a;
if any(I)
    tl=l(I); tu=u(I); x(I)=ntail(tl,tu);
end
% case 2: l<u<-a
J=u<-a;
if any(J)
    tl=-u(J); tu=-l(J); x(J)=-ntail(tl,tu);
end
% case 3: otherwise use inverse transform or accept-reject
I=~(I|J);
if  any(I)
    tl=l(I); tu=u(I); x(I)=tn(tl,tu);
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function x=ntail(l,u)
% samples a column vector of length=length(l)=length(u)
% from the standard multivariate normal distribution,
% truncated over the region [l,u], where l>0 and
% l and u are column vectors;
% uses acceptance-rejection from Rayleigh distr. 
% similar to Marsaglia (1964);
c=l.^2/2; n=length(l); f=expm1(c-u.^2/2);
x=c-reallog(1+rand(n,1).*f); % sample using Rayleigh
% keep list of rejected
I=find(rand(n,1).^2.*x>c); d=length(I);
while d>0 % while there are rejections
    cy=c(I); % find the thresholds of rejected
    y=cy-reallog(1+rand(d,1).*f(I));
    idx=rand(d,1).^2.*y<cy; % accepted
    x(I(idx))=y(idx); % store the accepted
    I=I(~idx); % remove accepted from list
    d=length(I); % number of rejected
end
x=sqrt(2*x); % this Rayleigh transform can be delayed till the end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function x=tn(l,u)
% samples a column vector of length=length(l)=length(u)
% from the standard multivariate normal distribution,
% truncated over the region [l,u], where -a<l<u<a for some
% 'a' and l and u are column vectors;
% uses acceptance rejection and inverse-transform method;
tol=2; % controls switch between methods
% threshold can be tuned for maximum speed for each platform
% case: abs(u-l)>tol, uses accept-reject from randn
I=abs(u-l)>tol; x=l;
if any(I)
    tl=l(I); tu=u(I); x(I)=trnd(tl,tu);
end
% case: abs(u-l)<tol, uses inverse-transform
I=~I;
if any(I)
    tl=l(I); tu=u(I); pl=erfc(tl/sqrt(2))/2; pu=erfc(tu/sqrt(2))/2;
    x(I)=sqrt(2)*erfcinv(2*(pl-(pl-pu).*rand(size(tl))));
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function  x=trnd(l,u)
% uses acceptance rejection to simulate from truncated normal
x=randn(size(l)); % sample normal
% keep list of rejected
I=find(x<l|x>u); d=length(I);
while d>0 % while there are rejections
    ly=l(I); % find the thresholds of rejected
    uy=u(I);
    y=randn(size(ly));
    idx=y>ly&y<uy; % accepted
    x(I(idx))=y(idx); % store the accepted
    I=I(~idx); % remove accepted from list
    d=length(I); % number of rejected
end
end




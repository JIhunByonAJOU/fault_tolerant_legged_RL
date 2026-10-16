"""Shared world-frame spatial path follower for both policies."""
import math
import torch


class CourseController:
    def __init__(self, protocol, rows, device):
        self.p=protocol
        self.device=device
        self.a=torch.tensor([r['profile']['amplitude'] for r in rows],device=device)
        self.yc=torch.tensor([(r['terrain_column_id']-.5)*8 for r in rows],device=device)
        self.fixed_yaw=torch.tensor([r['profile']['yaw'] or 0. for r in rows],device=device)
        self.sine=self.a>0
        self.k=2*math.pi/protocol['wavelength']
        self.x0=protocol['start_x']; self.x1=protocol['end_x']
        self.offsets=torch.linspace(-4,4,17,device=device)
        grid=torch.linspace(self.x0,self.x1,8193,device=device)
        self.dx=(self.x1-self.x0)/8192
        slope=self.a[:,None]*self.k*torch.cos(self.k*(grid-self.x0))[None,:]
        speed=torch.sqrt(1+slope*slope)
        self.arc=torch.cat((torch.zeros((len(rows),1),device=device),
                   torch.cumsum((speed[:,:-1]+speed[:,1:])*.5*self.dx,dim=1)),dim=1)
        self.last=None

    def evaluate(self, root):
        x,y=root[:,0],root[:,1]
        candidates=(x[:,None]+self.offsets).clamp(self.x0,self.x1)
        cy=self.yc[:,None]+self.a[:,None]*torch.sin(self.k*(candidates-self.x0))
        best=((candidates-x[:,None])**2+(cy-y[:,None])**2).argmin(dim=1)
        xp=candidates.gather(1,best[:,None]).squeeze(1)
        for _ in range(4):
            z=self.k*(xp-self.x0)
            yp=self.yc+self.a*torch.sin(z)
            d=self.a*self.k*torch.cos(z)
            dd=-self.a*self.k*self.k*torch.sin(z)
            delta=((xp-x)+(yp-y)*d)/(1+d*d+(yp-y)*dd).clamp_min(.25)
            xp=(xp-delta.clamp(-.5,.5)).clamp(self.x0,self.x1)
        z=self.k*(xp-self.x0)
        yp=self.yc+self.a*torch.sin(z)
        d=self.a*self.k*torch.cos(z)
        norm=torch.sqrt(1+d*d)
        cross=((y-yp)-d*(x-xp))/norm
        fractional=((xp-self.x0)/self.dx).clamp(0,8192)
        index=fractional.long().clamp(max=8191)
        arc0=self.arc.gather(1,index[:,None]).squeeze(1)
        arc1=self.arc.gather(1,(index+1)[:,None]).squeeze(1)
        s=arc0+(arc1-arc0)*(fractional-index)
        look_x=(xp+self.p['controller']['lookahead_m']/norm).clamp(max=self.x1)
        theta=torch.atan(self.a*self.k*torch.cos(self.k*(look_x-self.x0)))
        tangent=torch.stack((torch.cos(theta),torch.sin(theta)),dim=1)
        normal=torch.stack((-d/norm,1/norm),dim=1)
        world=tangent*self.p['speed']-self.p['controller']['lateral_gain']*cross[:,None]*normal
        world=world/torch.linalg.vector_norm(world,dim=1,keepdim=True).clamp_min(1e-8)*self.p['speed']
        q=root[:,3:7]
        yaw=torch.atan2(2*(q[:,3]*q[:,2]+q[:,0]*q[:,1]),1-2*(q[:,1]**2+q[:,2]**2))
        desired=torch.where(self.sine,theta,self.fixed_yaw)
        angle=torch.atan2(torch.sin(desired-yaw),torch.cos(desired-yaw))
        curvature=(-self.a*self.k*self.k*torch.sin(z))/(1+d*d)**1.5
        omega=(self.p['speed']*curvature+self.p['controller']['heading_gain']*angle).clamp(
            -self.p['controller']['max_yaw_rate'],self.p['controller']['max_yaw_rate'])
        c=torch.cos(yaw); ss=torch.sin(yaw)
        cmd=torch.stack((c*world[:,0]+ss*world[:,1],-ss*world[:,0]+c*world[:,1],omega),dim=1)
        self.last=dict(command=cmd, cross=cross, s=s, ref=torch.stack((xp,yp),dim=1),
                       yaw=yaw, goal=(x>=self.x1)&(cross.abs()<.5))
        return self.last

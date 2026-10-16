import math
import unittest
import torch

from legged_gym.evaluation.course_campaign import make_protocol, make_jobs, specs, conditions, binary_survival
from legged_gym.evaluation.course_controller import CourseController


class CourseTests(unittest.TestCase):
    def test_full_requested_grid_and_pairing(self):
        p=make_protocol(); jobs=make_jobs(p,'exploration')
        records={}; pair={}
        for job in jobs:
            for row in specs(p,'exploration',job['seed'],job['conditions']):
                key=row['paired_trial_id']+'_'+job['policy']
                self.assertNotIn(key,records); records[key]=row
                if row['paired_trial_id'] in pair: self.assertEqual(pair[row['paired_trial_id']],row)
                pair[row['paired_trial_id']]=row
        self.assertEqual(len(records),2928*12)
        self.assertEqual(len(jobs),18)
        self.assertTrue(max(j['num_envs'] for j in jobs)<=3264)
        self.assertEqual(sum(r['condition']['loss']==0 for r in records.values()),576)

    def test_forward_two_have_independent_initial_conditions(self):
        p=make_protocol(); rows=specs(p,'exploration',1101,conditions()[:1])
        self.assertNotEqual(rows[0]['initial_seed'],rows[12]['initial_seed'])
        self.assertEqual(rows[0]['profile']['yaw'],rows[12]['profile']['yaw'])

    def test_confirmation_keeps_run_budget_and_both_subprofiles(self):
        p=make_protocol(); units=[1,3,1,3]
        jobs=make_jobs(p,'confirmation',units)
        self.assertEqual(sum(j['num_envs'] for j in jobs),2928*12)
        for condition in conditions():
            present=set()
            for seed in p['confirmation_seeds']:
                rows=specs(p,'confirmation',seed,[condition],units)
                self.assertEqual(len(rows),96)
                present.update(r['profile']['id'] for r in rows)
            self.assertEqual(present,{r['id'] for r in p['profiles']})

    def test_binary_survival_is_independent_of_standing_duration(self):
        self.assertEqual(binary_survival('administrative_cap'),1)
        self.assertEqual(binary_survival('goal'),1)
        self.assertEqual(binary_survival('fall'),0)
        with self.assertRaises(ValueError): binary_survival('incomplete')
        self.assertNotIn('survival_horizon_s',make_protocol())

    def test_rotated_body_and_spatial_projection(self):
        p=make_protocol(); rows=specs(p,'exploration',1101,conditions()[:1])
        c=CourseController(p,rows,'cpu')
        root=torch.zeros((len(rows),13)); root[:,6]=1
        root[:,0]=5
        for i,row in enumerate(rows):
            a=row['profile']['amplitude']; k=2*math.pi/16
            root[i,1]=(row['terrain_column_id']-.5)*8+a*math.sin(k*6)
            yaw=row['profile']['yaw']
            if yaw is None: yaw=math.atan(a*k*math.cos(k*6))
            root[i,5]=math.sin(yaw/2); root[i,6]=math.cos(yaw/2)
        result=c.evaluate(root)
        self.assertLess(result['cross'].abs().max().item(),1e-4)
        self.assertTrue(torch.allclose(torch.linalg.vector_norm(result['command'][:,:2],dim=1),torch.full((len(rows),),.5),atol=1e-6))
        first=result['s'].clone(); same=c.evaluate(root)
        self.assertTrue(torch.equal(first,same['s']))


if __name__=='__main__': unittest.main()

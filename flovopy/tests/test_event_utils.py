"""Dataset-independent event utility tests; runnable with unittest or pytest."""
import unittest
import pandas as pd
from obspy.core.event import Event,Comment
from flovopy.core.event_utils import associate_events_by_time,parse_seed_id,get_comment_value

class EventUtilsTests(unittest.TestCase):
    def frames(self):
        t=pd.Timestamp('2000-01-01T00:00:00Z')
        left=pd.DataFrame({'lid':['001','002'],'lt':[t,t+pd.Timedelta(seconds=1)]})
        right=pd.DataFrame({'rid':['h1'],'rt':[t]})
        return left,right

    def match(self,left,right,**kwargs):
        return associate_events_by_time(left,right,left_id_col='lid',left_time_col='lt',
            right_id_col='rid',right_time_col='rt',**kwargs)

    def test_policies_and_input_immutability(self):
        left,right=self.frames();a=left.copy(deep=True);b=right.copy(deep=True)
        one=self.match(left,right,tolerance_seconds=2)
        nearest=self.match(left,right,tolerance_seconds=2,policy='nearest')
        self.assertEqual(list(one.left_id),['001']);self.assertEqual(len(nearest),2)
        self.assertEqual(list(one.columns),['left_id','left_time','right_id','right_time','abs_dt_s'])
        pd.testing.assert_frame_equal(left,a);pd.testing.assert_frame_equal(right,b)

    def test_boundary_and_empty(self):
        left,right=self.frames();right['rt']+=pd.Timedelta(seconds=3)
        self.assertEqual(len(self.match(left,right,tolerance_seconds=2)),1)
        self.assertTrue(self.match(left,right,tolerance_seconds=1.99).empty)
        for policy in ('one_to_one','nearest'):
            self.assertTrue(self.match(left.iloc[:0],right,policy=policy).empty)
            self.assertTrue(self.match(left,right.iloc[:0],policy=policy).empty)

    def test_timezones_missing_and_invalid(self):
        left,right=self.frames()
        left['lt']=['1999-12-31T19:00:00-05:00',None]
        right['rt']=['2000-01-01 00:00:00']
        self.assertEqual(self.match(left,right).iloc[0].abs_dt_s,0)
        left.loc[0,'lt']='bad'
        with self.assertRaises(ValueError):self.match(left,right)

    def test_ties(self):
        left,right=self.frames();left.loc[1,'lt']=left.loc[0,'lt']
        a=self.match(left,right);b=self.match(left.iloc[::-1],right)
        pd.testing.assert_frame_equal(a,b)
        left=left.iloc[:1];t=left.iloc[0]['lt']
        right=pd.DataFrame({'rid':['earlier','later'],'rt':[t-pd.Timedelta(seconds=1),t+pd.Timedelta(seconds=1)]})
        self.assertEqual(self.match(left,right,policy='nearest').iloc[0].right_id,'earlier')

    def test_large_integer_ids_and_zero_tolerance(self):
        left,right=self.frames();large=2**60+1
        right['rid']=[large]
        result=self.match(left,right,policy='nearest',tolerance_seconds=0)
        self.assertEqual(result.iloc[0].right_id,large)
        self.assertEqual(len(result),1)

    def test_validation(self):
        left,right=self.frames()
        for bad in (-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):self.match(left,right,tolerance_seconds=bad)
        with self.assertRaises(ValueError):self.match(left,right,policy='bad')
        with self.assertRaises(ValueError):self.match(left,right.drop(columns='rt'))
        left.loc[1,'lid']='001'
        with self.assertRaises(ValueError):self.match(left,right)
        left.loc[1,'lid']=None
        with self.assertRaises(ValueError):self.match(left,right)

    def test_seed_ids(self):
        self.assertEqual(parse_seed_id('XB.PI2..ELZ'),('XB','PI2','','ELZ'))
        self.assertEqual(parse_seed_id('.PI2..ELZ'),('','PI2','','ELZ'))
        for value in (None,'','bad','XB.PI2.ELZ',float('nan')):
            self.assertEqual(parse_seed_id(value),(None,None,None,None))

    def test_comments(self):
        event=Event(comments=[Comment(text=' event_class: W_P '),Comment(text='event_class:H_ONLY')])
        self.assertEqual(get_comment_value(event,'event_class:'),'W_P')
        self.assertIsNone(get_comment_value(event,'EVENT_CLASS:'))
        self.assertIsNone(get_comment_value(Event(),'event_class:'))
        with self.assertRaises(ValueError):get_comment_value(event,'')

if __name__=='__main__':unittest.main()

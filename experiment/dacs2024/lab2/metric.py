import os
import re

class MetricParser():
    def __init__(self, rundir):
        self.rundir = rundir

        self.area_rpt = os.path.join(rundir, 'innovus-rundir/reports/floorplan_area.rpt')
        self.drv_rpt = os.path.join(rundir, 'innovus-rundir/reports/chipdone_drv.rpt')
        self.power_rpt = os.path.join(rundir, 'innovus-rundir/reports/chipdone_static_power.rpt')
        self.setup_timing_rpt = os.path.join(rundir, 'innovus-rundir/reports/chipdone_setup_slack_summary.rpt')
        self.hold_timing_rpt = os.path.join(rundir, 'innovus-rundir/reports/chipdone_hold_slack_summary.rpt')
        
        # These will be the stored results
        self.area = None
        self.drv = None
        self.power = None
        self.freq = None
        self.setup_slack = None
        self.hold_slack = None

    def parse_area(self):
        with open(self.area_rpt, 'r') as f:
            self.area = float(f.readlines()[0])

    def parse_drv(self):
        with open(self.drv_rpt, 'r') as f:
            for line in f.readlines():
                if line.startswith('  Total Violations'):
                    self.drv = float(re.findall(r'\d+', line)[0])
                if line.startswith('No DRC violations were found'):
                    self.drv = 0

    def parse_power(self):
        with open(self.power_rpt, 'r') as f:
            for line in f.readlines():
                if line.startswith('Total Power:'):
                    self.power = float(re.findall(r'\d+\.?\d*', line)[0])

    def parse_setup_timing(self):
        with open(self.setup_timing_rpt, 'r') as f:
            for line in f.readlines():
                if line.startswith('Worst negative slacks(WNS)'):
                    self.setup_slack = float(re.findall(r'-?\d+\.?\d*', line)[0])

    def parse_hold_timing(self):
        with open(self.hold_timing_rpt, 'r') as f:
            for line in f.readlines():
                if line.startswith('Worst negative slacks(WNS)'):
                    self.hold_slack = float(re.findall(r'-?\d+\.?\d*', line)[0])

    def generate_report(self):
        self.parse_area()
        self.parse_drv()
        self.parse_power()
        self.parse_setup_timing()
        self.parse_hold_timing()
        reports = {
            'area(um^2)': self.area,
            'power(mW)': self.power,
            '#drv': self.drv,
            'setup_slack(ns)': self.setup_slack,
            'hold_slack(ns)': self.hold_slack
        }
        return reports

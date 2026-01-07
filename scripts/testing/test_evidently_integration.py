#!/usr/bin/env python3
"""
Test script for Test scripi for Evide.
tly a data dr ft,ifargt tdrirt,gaid and dqualaqyuality monituuontityal.y.
"""

import sys
from sythlib import Path
import pas
from pathlib import Path
imporemedeme

# Add pmpjectr poa ts path
p jct=h(__fele__).pameottp tene
ime.th.ist(0sr(ojct_oo))

# Add protaoa dtft_mitor
project_root = PDafaDr_ftMopiaor crassnwtph sampla dnta.
sys.path.in🧪rt(0, strDaDrftMitor...

def try:test_data_drift_monitor():
    "ehtiromtsrc.monitor cl.sapa_drift_mogtoraDmportnitor.."), DRept,TrgeRept
        ng.data_drift_monitor import DataDriftMonitor, DriftReport, TargetDriftReport
      int✅ DD  pMnt("✅  imporis succcssful"l")
            
            # Crealmpd estin fo mtesing
            n_samples = 1000
        n_samples = 1000    
        
          Refe eneredata (trac d(g dataransnr batst))
    dt  r fe.Dmce_dae= pd.DaFam({
      g     'age': ap.'anromon.rral(40, 10, m_aamples),(50000, 20000, n_samples),
            'bala'pe': np.r.rdom.noomalrm0000, 2000a,(0_s1mpl0e),,
            'dcm':i.s':3010smples,
            'campaig ': np. aniom.po ssnn(2, p_samples),.random.choice(['married', 'single', 'divorced'], n_samples),
            'jeb':tnp.oa.dnmooho[iy(['admi''on'blua-cylla'', ',ech'icitn', 'sarryc]s'], ,_sampn,,
        'prd'mart'a:':nnp..andom.chorcc(['married',[aspngle',e's v52'],n_sampl,
       'eduatin'np.ranom.choice(['pmay', 'scnday', 'etary'], nampls,
       'chr'anom.hice([0, 1], _ampls, =[0.8, 02],
            'pre  cuion': np.randtmacho cl([0, 1], ngsampler, p=[0.75, 0.25])
        })fted)
        
    dat #taarreetata (slghly 
    'ag'cur.dot_dnoa =lp4.DaaaF)#me({ted mean and std
            'ag ':dnp.o':drm.aommal(45, 12, n_samples),  # D  f ad mpomoa d_s)dfted mean
            'balance': n .  'do'. .rmal(45000,a25000,dn_samphis),  # D[ f'ed-collar', 'technician', 'services'], n_samples, p=[0.3, 0.3, 0.2, 0.2]),
          'a'duia np.': churrndom.nonm:ln28.ran2m.cn_samples)oi(#,Drifted
]nal,=. 3,#r'campaign': np.random.poisson(3ftn_samples ,  # D ift   mean
'           'job': preitndom.choiceon'admin': 'blue-collar'p.'technician'an'services']omn_sampleschp=[1.3],s.3ams.2, p.2])=
[6 3)#      'ma itl':ndom.choice'married''single''divorced'n_samplesp=3)
        tet 'education': nprrandom.choice(['primary''secondary''tertiary'n_samplesp=[53)
            'churn': np.random.choice(moto1 Dan_samplesifp=or(73)#Driftedtarget
            'prediction': nprandomchoice(1n_samplesp=655)  #Driftedpredictions
        }   # Set reference data
            monitor.set_reference_data(reference_data)
          Iniria✅izeferencsecessfully")
      r = DataDiftM()
        
        # St eee data
        moniToe.sttt efdeenct_eata(iefrcata    drift_report = monitor.detect_drift(current_data, "test_drift_report")
            prn" R refesco datars:  sucresifuroyri
        ft_score:.3f}")
        # Test data d ift detect o 
    t(f driDt_repirt =tmodi or.dcmecn_d(dft(rurrpnt_oata,r"test_drift_repert_columns)}")
            ✅Datadrifttct completeddrift_.ft_detected
            # Test Driftasgo eft ddift_retect.ditoe
            try:Driftedlumnslen(dift_r.ditdcoln)
                target_report = monitor.detect_target_drift(reference_data, current_data, "test_target_drift")
        #    prta(gef"drifaededrc detection completed")
    nT  aey:arget_drift_detected}")
            t  ge  report =  opitor.detect_trrint_dtift(r fePence_dard, curreit_dcta, "test_tartit_doift" drift: {target_report.prediction_drift_detected}")
         p( p️in (a"✅eTargedift dettntectiad  ompl(ted")expected for Evidently 0.4.0): {str(e)}")
            p  n(f"   Trgt dif:{tgerpottagdif_detected}")
            print(f"   PrTdiction deifs: { argtt_rupoal.yr dictiondrift_detected}")
        uxclpy Excuption as  :n_data_quality_check(current_data, "test_quality_check")
            print(f"⚠️  Targep drrft deteciion fained (expecttd for "✅ Data q 0.4.0):u{sic(o)}l)
        ted")
        # Trnt dPassqual eysch {k
        qualiqy_rusults = monitor.run_aata_qualtyy_c_eck(surltnt_data,s"test_qualasyech_ck")
        t]}nt(/"✅ Data quali{y check almplettdu)ts['total_tests']}")
        print(f"   Passed test: {qality_sults['pass_tss']}/{quaitresuls['tals']})
        
        # T st history funcTiontli ytory functionality
    histh s ity =gmon_rof.gts_ory(5)hity(5)
    prinpf"nt(✅"✅ History re riHvarri{len(has:ory)} {len(hs reotd")found")
        
        return True
        
    except Exception as e:
        print(f"❌ DataDriftMonitor test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Additional debugging: try to create a simple report manually
        try:
            print("\n🔍 Additional debugging: Creating simple report manually...")
            from evidently.report import Report
            from evidently.metrics import DatasetSummaryMetric
            from evidently.pipeline.column_mapping import ColumnMapping
            
            # Create simple test data
            simple_ref = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
            simple_curr = pd.DataFrame({'feature1': [2, 3, 4], 'feature2': [5, 6, 7]})
            
            # Create column mapping
            col_map = ColumnMapping(numerical_features=['feature1', 'feature2'])
            
            # Create and run simple report
            simple_report = Report(metrics=[DatasetSummaryMetric()])
            simple_report.run(reference_data=simple_ref, current_data=simple_curr, column_mapping=col_map)
            print("✅ Simple report creation successful")
            
        except Exception as debug_e:
            print(f"❌ Simple report debugging failed: {str(debug_e)}")
            import traceback
            traceback.print_exc()
        
        return False

def test_evidently_imports():
    """Test that all Evidently imports work correctly."""
    print("\n🧪 Testing Evidently imports...")
    
    try:
        # Test Evidently core imports
        from evidently.report import Report
        from evidently.metrics import (
            DataDriftTable, DatasetDriftMetric, DatasetMissingValuesMetric,
            DatasetSummaryMetric, ColumnDriftMetric
        )
        from evidently.test_suite import TestSuite
        from evidently.tests import TestNumberOfColumnsWithMissingValues
        
        print("✅ All Evidently imports successful")
        
        # Test creating a simple report
        sample_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5]
        })
        
        report = Report(metrics=[DatasetSummaryMetric()])
        report.run(reference_data=sample_data, current_data=sample_data)
        
        print("✅ Evidently report creation successful")
        return True
        
    except Exception as e:
        print(f"❌ Evidently imports test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_configuration_loading():
    """Test that the monitoring configuration loads correctly."""
    print("\n🧪 Testing configuration loading...")
    
    try:
        from src.utils.config_loader import load_config
        
        config = load_config("configs/monitoring/monitoring_config.yaml")
        
        # Check Evidently-specific configuration
        evidently_config = config['data_drift_monitoring'].get('evidently', {})
        
        if evidently_config:
            print("✅ Evidently configuration found")
            print(f"   Target column: {evidently_config.get('target_column')}")
            print(f"   Prediction column: {evidently_config.get('prediction_column')}")
            print(f"   Task type: {evidently_config.get('task_type')}")
        else:
            print("⚠️  No Evidently-specific configuration found")
            
        return True
        
    except Exception as e:
        print(f"❌ Configuration loading test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_alert_integration():
    """Test integration with alert system."""
    print("\n🧪 Testing alert integration...")
    
    try:
        from src.monitoring.alert_system import AlertManager, AlertType, AlertSeverity
        
        # Test that alert system can be imported and initialized
        alert_manager = AlertManager()
        print("✅ AlertManager initialization successful")
        
        # Test creating a drift alert
        alert = alert_manager.create_alert(
            alert_type=AlertType.DATA_DRIFT,
            severity=AlertSeverity.HIGH,
            title="Test Data Drift Alert",
            message="This is a test alert for data drift detection",
            source="test_evidently_integration"
        )
        
        print(f"✅ Alert creation successful: {alert.id}")
        return True
        
    except Exception as e:
        print(f"❌ Alert integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all Evidently integration tests."""
    print("🚀 Running Evidently Integration Tests...")
    print("=" * 60)
    
    tests = [
        ("Evidently Imports", test_evidently_imports),
        ("Configuration Loading", test_configuration_loading),
        ("Data Drift Monitor", test_data_drift_monitor),
        ("Alert Integration", test_alert_integration),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            print(f"\n{test_name}...")
            result = test_func()
            results[test_name] = result
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status} {test_name}")
        except Exception as e:
            print(f"❌ ERROR in {test_name}: {str(e)}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 EVIDENTLY INTEGRATION TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Evidently integration tests passed!")
        print("\n📋 Next steps:")
        print("1. Run: python scripts/run_monitoring.py --test-evidently")
        print("2. Check generated reports in reports/drift_reports/")
        print("3. Verify integration with alert system")
        return True
    else:
        print("⚠️  Some Evidently integration tests failed")
        print("   Check the errors above and ensure Evidently is properly installed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

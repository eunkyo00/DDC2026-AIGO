"""Synthetic split invariants; no audio, external data or model required."""
import unittest
from create_split import assign, check_partitions, sha, validate_calls


def fixture():
    return [dict(call_id=sha(f'{g}{i}'),gender=g,wav_path=f'/synthetic/{g}{i}.wav',
                 wav_relative_path=f'Training/{g}{i}.wav',_prefix=sha(f'prefix-{g}{i}'))
            for g in ('M','F') for i in range(10)]


class SplitTests(unittest.TestCase):
    def test_stratification_and_order_invariance(self):
        rows=fixture();a=assign(rows)
        self.assertEqual(a,assign(list(reversed(rows))))
        for gender in ('M','F'):
            self.assertEqual(sum(a[r['call_id']]=='internal_validation' for r in rows if r['gender']==gender),2)
        self.assertNotEqual(a,assign(rows,43))
        train,val,checks=check_partitions(rows,a)
        self.assertEqual((len(train),len(val)),(16,4))
        self.assertTrue(all(v==0 for v in checks.values()))

    def test_duplicate_conflicting_and_missing_gender_fail(self):
        rows=fixture()
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            validate_calls(rows+[dict(rows[0])])
        with self.assertRaisesRegex(ValueError,'Conflicting'):
            validate_calls(rows+[dict(rows[0],gender='F')])
        with self.assertRaisesRegex(ValueError,'gender'):
            validate_calls([dict(rows[0],gender='')])
        with self.assertRaisesRegex(ValueError,'Expected'):
            validate_calls(rows,21)

    def test_leakage_and_partial_membership_fail(self):
        rows=fixture();a=assign(rows)
        tr=next(r for r in rows if a[r['call_id']]=='train')
        va=next(r for r in rows if a[r['call_id']]=='internal_validation')
        va['wav_path']=tr['wav_path']
        with self.assertRaisesRegex(ValueError,'wav_path_overlap'):
            check_partitions(rows,a)
        rows=fixture();a.pop(rows[0]['call_id'])
        with self.assertRaisesRegex(ValueError,'cover'):
            check_partitions(rows,a)

    def test_additional_source_id_crossing_fails(self):
        rows=fixture();a=assign(rows)
        tr=next(r for r in rows if a[r['call_id']]=='train')
        va=next(r for r in rows if a[r['call_id']]=='internal_validation')
        va['_prefix']=tr['_prefix']
        with self.assertRaisesRegex(ValueError,'filename_prefix_overlap'):
            check_partitions(rows,a)


if __name__=='__main__':
    unittest.main()

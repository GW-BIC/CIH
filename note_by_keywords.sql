select a.PatientICN, b.TIUDocumentSID, b.ReferenceDateTime, RandID = NEWID()
	into Dflt.YS_TIU_ACUP
from Src.SPatient_SPatient as a
join Src.TIU_TIUDocument as b
on a.PatientSID = b.PatientSID
join Src.tvf_TIU_FullTextSearch('acupuncture OR acup') as c
on b.TIUDocumentSID = c.TIUDocumentSID

select a.PatientICN, b.TIUDocumentSID, b.ReferenceDateTime, RandID = NEWID()
	into Dflt.YS_TIU_BIOF
from Src.SPatient_SPatient as a
join Src.TIU_TIUDocument as b
on a.PatientSID = b.PatientSID
join Src.tvf_TIU_FullTextSearch('biofeedback') as c
on b.TIUDocumentSID = c.TIUDocumentSID

select a.PatientICN, b.TIUDocumentSID, b.ReferenceDateTime, RandID = NEWID()
	into Dflt.YS_TIU_GUID
from Src.SPatient_SPatient as a
join Src.TIU_TIUDocument as b
on a.PatientSID = b.PatientSID
join Src.tvf_TIU_FullTextSearch('"guided imagery" OR "guided visualization"') as c
on b.TIUDocumentSID = c.TIUDocumentSID

select a.PatientICN, b.TIUDocumentSID, b.ReferenceDateTime, RandID = NEWID()
	into Dflt.YS_TIU_MEDI
from Src.SPatient_SPatient as a
join Src.TIU_TIUDocument as b
on a.PatientSID = b.PatientSID
join Src.tvf_TIU_FullTextSearch('meditation OR mindfulness OR mantram OR mbsr OR mbct') as c
on b.TIUDocumentSID = c.TIUDocumentSID

select a.PatientICN, b.TIUDocumentSID, b.ReferenceDateTime, RandID = NEWID()
	into Dflt.YS_TIU_TAIC
from Src.SPatient_SPatient as a
join Src.TIU_TIUDocument as b
on a.PatientSID = b.PatientSID
join Src.tvf_TIU_FullTextSearch('"Tai Chi" OR "T''ai Chi" OR TaiChi OR "Qi Gong" OR QiGong') as c
on b.TIUDocumentSID = c.TIUDocumentSID

select a.PatientICN, b.TIUDocumentSID, b.ReferenceDateTime, RandID = NEWID()
	into Dflt.YS_TIU_YOGA
from Src.SPatient_SPatient as a
join Src.TIU_TIUDocument as b
on a.PatientSID = b.PatientSID
join Src.tvf_TIU_FullTextSearch('yoga OR "breathing stretching relaxation" OR pranayama OR vinyasa OR hatha') as c
on b.TIUDocumentSID = c.TIUDocumentSID
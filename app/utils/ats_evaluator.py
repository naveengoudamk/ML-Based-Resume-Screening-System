
import re

class ATSEvaluator:
    def __init__(self, resume_text, jd_text):
        self.resume_text = resume_text.lower()
        self.jd_text = jd_text.lower()
        self.raw_text = resume_text # Case sensitive for some checks if needed
        
    def check_contact_info(self):
        # Email Regex
        email = re.search(r'[\w\.-]+@[\w\.-]+', self.resume_text)
        # Phone Regex (Simple)
        phone = re.search(r'(\d{10})|(\+\d{1,2}\s?\d{10})', self.resume_text.replace("-", "").replace(" ", ""))
        # LinkedIn/Links
        linkedin = "linkedin.com" in self.resume_text
        
        return {
            'email': bool(email),
            'phone': bool(phone),
            'linkedin': bool(linkedin)
        }
        
    def check_sections(self):
        # Common headers
        sections = {
            'education': ['education', 'academic', 'qualification'],
            'experience': ['experience', 'work history', 'employment'],
            'skills': ['skills', 'technologies', 'competencies', 'expertise'],
            'projects': ['projects', 'initiatives']
        }
        
        found_sections = {}
        for sec, keywords in sections.items():
            found = any(k in self.resume_text for k in keywords)
            found_sections[sec] = found
            
        return found_sections
        
    def check_action_verbs(self):
        # Strong action verbs often checked by ATS/ResumeWorded
        verbs = [
            "developed", "designed", "implemented", "managed", "led", 
            "created", "achieved", "improved", "increased", "resolved",
            "collaborated", "orchestrated", "engineered", "optimized"
        ]
        
        count = sum(1 for v in verbs if v in self.resume_text)
        score = min(count * 5, 100) # Cap at 20 verbs
        return score, count
        
    def calculate_keyword_match(self):
        # Extract keywords from JD (simple length filter > 3)
        jd_words = set(re.findall(r'\b[a-z]{4,}\b', self.jd_text))
        resume_words = set(re.findall(r'\b[a-z]{4,}\b', self.resume_text))
        
        if not jd_words:
            return 0, []
        
        common = jd_words.intersection(resume_words)
        missing = list(jd_words - resume_words)[:10] # Top 10 missing
        
        match_score = (len(common) / len(jd_words)) * 100
        return round(match_score, 1), missing

    def evaluate(self):
        contact = self.check_contact_info()
        sections = self.check_sections()
        verb_score, verb_count = self.check_action_verbs()
        keyword_score, missing_keywords = self.calculate_keyword_match()
        
        # Calculate Weighted ATS Score
        # Structure (Contact + Sections): 25%
        # Impact (Verbs): 15%
        # Content (Keywords): 60%
        
        structure_points = sum(contact.values()) + sum(sections.values())
        max_structure = len(contact) + len(sections)
        structure_score = (structure_points / max_structure) * 100
        
        final_score = (structure_score * 0.25) + (verb_score * 0.15) + (keyword_score * 0.60)
        
        return {
            'overall_score': round(final_score, 1),
            'detail_scores': {
                'structure': round(structure_score, 1),
                'impact': round(verb_score, 1),
                'keywords': round(keyword_score, 1)
            },
            'checks': {
                'contact': contact,
                'sections': sections,
                'verb_count': verb_count
            },
            'missing_keywords': missing_keywords
        }

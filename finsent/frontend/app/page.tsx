// =============================================================================
// LANDING PAGE
// =============================================================================
// Main landing page composing all landing sections
//
// Location: frontend/app/page.tsx
//
// =============================================================================

import {
  Hero,
  TrustedBy,
  Features,
  HowItWorks,
  Pricing,
  Testimonials,
  FAQ,
  FinalCTA,
} from '@/components/landing';

export default function LandingPage() {
  return (
    <>
      <Hero />
      <TrustedBy />
      <Features />
      <HowItWorks />
      <Pricing />
      <Testimonials />
      <FAQ />
      <FinalCTA />
    </>
  );
}

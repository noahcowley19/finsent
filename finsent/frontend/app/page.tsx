// =============================================================================
// LANDING PAGE - Financial Operating System
// =============================================================================
// Main landing page with Command Bar navigation and Bento Grid layout
//
// Location: frontend/app/page.tsx
//
// =============================================================================

import { HeroCommand, BentoGrid, PricingDark, FAQDark, FinalCTADark } from '@/components/landing';

export default function LandingPage() {
  return (
    <div className="bg-obsidian-950 -mt-16">
      <HeroCommand />
      <BentoGrid />
      <PricingDark />
      <FAQDark />
      <FinalCTADark />
    </div>
  );
}

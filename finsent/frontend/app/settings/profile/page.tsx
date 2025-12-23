'use client';

// =============================================================================
// PROFILE SETTINGS PAGE
// =============================================================================
// User profile and password management
//
// Location: frontend/app/settings/profile/page.tsx
//
// =============================================================================

import React, { useState } from 'react';
import { useAuth } from '@/lib/auth-context';

export default function ProfileSettingsPage() {
  const { user } = useAuth();
  
  const [profileData, setProfileData] = useState({
    name: user?.name || '',
    email: user?.email || '',
  });
  const [passwordData, setPasswordData] = useState({
    currentPassword: '',
    newPassword: '',
    confirmPassword: '',
  });
  const [isSavingProfile, setIsSavingProfile] = useState(false);
  const [isSavingPassword, setIsSavingPassword] = useState(false);
  const [profileMessage, setProfileMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
  const [passwordMessage, setPasswordMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  const handleProfileSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSavingProfile(true);
    setProfileMessage(null);

    try {
      // TODO: Implement API call
      await new Promise((resolve) => setTimeout(resolve, 1000));
      setProfileMessage({ type: 'success', text: 'Profile updated successfully' });
    } catch (error) {
      setProfileMessage({ type: 'error', text: 'Failed to update profile' });
    } finally {
      setIsSavingProfile(false);
    }
  };

  const handlePasswordSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSavingPassword(true);
    setPasswordMessage(null);

    if (passwordData.newPassword !== passwordData.confirmPassword) {
      setPasswordMessage({ type: 'error', text: 'Passwords do not match' });
      setIsSavingPassword(false);
      return;
    }

    try {
      // TODO: Implement API call
      await new Promise((resolve) => setTimeout(resolve, 1000));
      setPasswordMessage({ type: 'success', text: 'Password changed successfully' });
      setPasswordData({ currentPassword: '', newPassword: '', confirmPassword: '' });
    } catch (error) {
      setPasswordMessage({ type: 'error', text: 'Failed to change password' });
    } finally {
      setIsSavingPassword(false);
    }
  };

  return (
    <div className="space-y-8">
      {/* Profile section */}
      <div className="bg-white rounded-xl border border-border-light p-6">
        <h2 className="font-heading font-semibold text-heading-md text-navy-900 mb-6">
          Profile Information
        </h2>

        <form onSubmit={handleProfileSubmit} className="space-y-6">
          {/* Avatar */}
          <div className="flex items-center gap-6">
            <div className="w-20 h-20 rounded-full bg-gradient-to-br from-terra-400 to-terra-600 flex items-center justify-center text-white text-2xl font-semibold">
              {user?.name?.charAt(0) || user?.email?.charAt(0).toUpperCase()}
            </div>
            <div>
              <button
                type="button"
                className="px-4 py-2 bg-white border border-border-medium rounded-lg text-body-sm font-medium text-navy-700 hover:bg-cream-50 transition-colors"
              >
                Change Avatar
              </button>
              <p className="text-caption text-neutral-500 mt-1">
                JPG, PNG or GIF. Max 2MB.
              </p>
            </div>
          </div>

          {/* Name */}
          <div>
            <label htmlFor="name" className="block text-body-sm font-medium text-navy-700 mb-2">
              Full Name
            </label>
            <input
              id="name"
              type="text"
              value={profileData.name}
              onChange={(e) => setProfileData({ ...profileData, name: e.target.value })}
              className="w-full h-12 px-4 bg-white border border-border-medium rounded-lg text-navy-900 focus:outline-none focus:border-navy-500 focus:ring-2 focus:ring-navy-500/20"
            />
          </div>

          {/* Email */}
          <div>
            <label htmlFor="email" className="block text-body-sm font-medium text-navy-700 mb-2">
              Email Address
            </label>
            <input
              id="email"
              type="email"
              value={profileData.email}
              onChange={(e) => setProfileData({ ...profileData, email: e.target.value })}
              className="w-full h-12 px-4 bg-white border border-border-medium rounded-lg text-navy-900 focus:outline-none focus:border-navy-500 focus:ring-2 focus:ring-navy-500/20"
            />
          </div>

          {/* Message */}
          {profileMessage && (
            <div
              className={`p-4 rounded-lg ${
                profileMessage.type === 'success'
                  ? 'bg-success-50 text-success-700'
                  : 'bg-error-50 text-error-700'
              }`}
            >
              {profileMessage.text}
            </div>
          )}

          {/* Submit */}
          <div className="flex justify-end">
            <button
              type="submit"
              disabled={isSavingProfile}
              className="px-6 py-2.5 bg-terra-500 text-white font-medium rounded-lg hover:bg-terra-600 transition-colors disabled:opacity-50"
            >
              {isSavingProfile ? 'Saving...' : 'Save Changes'}
            </button>
          </div>
        </form>
      </div>

      {/* Password section */}
      <div className="bg-white rounded-xl border border-border-light p-6">
        <h2 className="font-heading font-semibold text-heading-md text-navy-900 mb-6">
          Change Password
        </h2>

        <form onSubmit={handlePasswordSubmit} className="space-y-6">
          {/* Current password */}
          <div>
            <label htmlFor="currentPassword" className="block text-body-sm font-medium text-navy-700 mb-2">
              Current Password
            </label>
            <input
              id="currentPassword"
              type="password"
              value={passwordData.currentPassword}
              onChange={(e) => setPasswordData({ ...passwordData, currentPassword: e.target.value })}
              className="w-full h-12 px-4 bg-white border border-border-medium rounded-lg text-navy-900 focus:outline-none focus:border-navy-500 focus:ring-2 focus:ring-navy-500/20"
            />
          </div>

          {/* New password */}
          <div>
            <label htmlFor="newPassword" className="block text-body-sm font-medium text-navy-700 mb-2">
              New Password
            </label>
            <input
              id="newPassword"
              type="password"
              value={passwordData.newPassword}
              onChange={(e) => setPasswordData({ ...passwordData, newPassword: e.target.value })}
              className="w-full h-12 px-4 bg-white border border-border-medium rounded-lg text-navy-900 focus:outline-none focus:border-navy-500 focus:ring-2 focus:ring-navy-500/20"
            />
          </div>

          {/* Confirm password */}
          <div>
            <label htmlFor="confirmPassword" className="block text-body-sm font-medium text-navy-700 mb-2">
              Confirm New Password
            </label>
            <input
              id="confirmPassword"
              type="password"
              value={passwordData.confirmPassword}
              onChange={(e) => setPasswordData({ ...passwordData, confirmPassword: e.target.value })}
              className="w-full h-12 px-4 bg-white border border-border-medium rounded-lg text-navy-900 focus:outline-none focus:border-navy-500 focus:ring-2 focus:ring-navy-500/20"
            />
          </div>

          {/* Message */}
          {passwordMessage && (
            <div
              className={`p-4 rounded-lg ${
                passwordMessage.type === 'success'
                  ? 'bg-success-50 text-success-700'
                  : 'bg-error-50 text-error-700'
              }`}
            >
              {passwordMessage.text}
            </div>
          )}

          {/* Submit */}
          <div className="flex justify-end">
            <button
              type="submit"
              disabled={isSavingPassword}
              className="px-6 py-2.5 bg-navy-900 text-white font-medium rounded-lg hover:bg-navy-800 transition-colors disabled:opacity-50"
            >
              {isSavingPassword ? 'Changing...' : 'Change Password'}
            </button>
          </div>
        </form>
      </div>

      {/* Danger zone */}
      <div className="bg-white rounded-xl border border-error-200 p-6">
        <h2 className="font-heading font-semibold text-heading-md text-error-700 mb-2">
          Danger Zone
        </h2>
        <p className="text-body-sm text-neutral-600 mb-4">
          Once you delete your account, there is no going back. Please be certain.
        </p>
        <button
          type="button"
          className="px-4 py-2 bg-white border border-error-300 text-error-600 font-medium rounded-lg hover:bg-error-50 transition-colors"
        >
          Delete Account
        </button>
      </div>
    </div>
  );
}

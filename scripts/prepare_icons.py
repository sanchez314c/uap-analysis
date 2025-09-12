#!/usr/bin/env python3
"""
Icon Preparation Script for Cross-Platform Builds
Creates platform-specific icon files from source assets
"""

import os
import sys
import subprocess
from pathlib import Path
from PIL import Image

class IconPreparer:
    def __init__(self):
        self.project_dir = Path(__file__).parent.parent
        self.assets_dir = self.project_dir / "assets"
        self.icons_dir = self.assets_dir / "icons"
        
        # Icon specifications for different platforms
        self.icon_sizes = {
            'macos': [16, 32, 128, 256, 512, 1024],
            'windows': [16, 32, 48, 64, 128, 256],
            'linux': [16, 22, 24, 32, 48, 64, 128, 256, 512]
        }
        
    def setup_directories(self):
        """Create necessary directories"""
        self.icons_dir.mkdir(parents=True, exist_ok=True)
        
        platform_dirs = ['macos', 'windows', 'linux']
        for platform in platform_dirs:
            (self.icons_dir / platform).mkdir(exist_ok=True)
            
    def find_source_icon(self):
        """Find the source icon file"""
        # Look for common icon formats
        icon_patterns = ['*.icns', '*.png', '*.ico', '*.svg']
        
        for pattern in icon_patterns:
            icons = list(self.assets_dir.glob(pattern))
            if icons:
                print(f"✅ Found source icon: {icons[0]}")
                return icons[0]
                
        print("❌ No source icon found. Please place an icon file (.icns, .png, .ico, or .svg) in the assets directory")
        return None
        
    def create_macos_icns(self, source_icon):
        """Create macOS .icns file"""
        print("🍎 Creating macOS .icns file...")
        
        # If source is already .icns, copy it
        if source_icon.suffix.lower() == '.icns':
            icns_path = self.icons_dir / 'macos' / 'app_icon.icns'
            icns_path.write_bytes(source_icon.read_bytes())
            print(f"✅ Copied existing .icns file to {icns_path}")
            return icns_path
            
        # Create iconset directory
        iconset_dir = self.icons_dir / 'macos' / 'app_icon.iconset'
        iconset_dir.mkdir(exist_ok=True)
        
        try:
            # Load source image
            img = Image.open(source_icon)
            
            # Generate all required sizes
            for size in self.icon_sizes['macos']:
                # Standard resolution
                resized = img.resize((size, size), Image.Resampling.LANCZOS)
                resized.save(iconset_dir / f'icon_{size}x{size}.png')
                
                # High resolution (@2x)
                if size <= 512:  # Don't create @2x for sizes > 512
                    resized_2x = img.resize((size * 2, size * 2), Image.Resampling.LANCZOS)
                    resized_2x.save(iconset_dir / f'icon_{size}x{size}@2x.png')
                    
            # Convert to .icns using iconutil (macOS only)
            if sys.platform == 'darwin':
                icns_path = self.icons_dir / 'macos' / 'app_icon.icns'
                subprocess.run(['iconutil', '-c', 'icns', str(iconset_dir), '-o', str(icns_path)])
                print(f"✅ Created .icns file: {icns_path}")
                return icns_path
            else:
                print("⚠️  iconutil not available (not on macOS). Using PNG files for macOS build.")
                return iconset_dir
                
        except Exception as e:
            print(f"❌ Error creating macOS icons: {e}")
            return None
            
    def create_windows_ico(self, source_icon):
        """Create Windows .ico file"""
        print("🪟 Creating Windows .ico file...")
        
        try:
            img = Image.open(source_icon)
            
            # Create multiple sizes for ICO
            ico_images = []
            for size in self.icon_sizes['windows']:
                resized = img.resize((size, size), Image.Resampling.LANCZOS)
                ico_images.append(resized)
                
            # Save as ICO
            ico_path = self.icons_dir / 'windows' / 'app_icon.ico'
            ico_images[0].save(
                ico_path,
                format='ICO',
                sizes=[(size, size) for size in self.icon_sizes['windows']]
            )
            
            print(f"✅ Created .ico file: {ico_path}")
            return ico_path
            
        except Exception as e:
            print(f"❌ Error creating Windows icon: {e}")
            return None
            
    def create_linux_icons(self, source_icon):
        """Create Linux icon files"""
        print("🐧 Creating Linux icon files...")
        
        try:
            img = Image.open(source_icon)
            
            created_icons = []
            for size in self.icon_sizes['linux']:
                resized = img.resize((size, size), Image.Resampling.LANCZOS)
                icon_path = self.icons_dir / 'linux' / f'app_icon_{size}x{size}.png'
                resized.save(icon_path)
                created_icons.append(icon_path)
                
            print(f"✅ Created {len(created_icons)} Linux icon files")
            return created_icons
            
        except Exception as e:
            print(f"❌ Error creating Linux icons: {e}")
            return None
            
    def prepare_all_icons(self):
        """Prepare icons for all platforms"""
        print("🎨 Preparing icons for all platforms...")
        
        self.setup_directories()
        
        source_icon = self.find_source_icon()
        if not source_icon:
            return False
            
        # Create platform-specific icons
        results = {
            'macos': self.create_macos_icns(source_icon),
            'windows': self.create_windows_ico(source_icon),
            'linux': self.create_linux_icons(source_icon)
        }
        
        # Report results
        success_count = sum(1 for result in results.values() if result)
        print(f"\n🎯 Icon preparation complete: {success_count}/3 platforms successful")
        
        return success_count > 0

if __name__ == "__main__":
    preparer = IconPreparer()
    success = preparer.prepare_all_icons()
    
    if success:
        print("✅ Icon preparation completed successfully!")
        sys.exit(0)
    else:
        print("❌ Icon preparation failed!")
        sys.exit(1)
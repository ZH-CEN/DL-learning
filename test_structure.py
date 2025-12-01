"""
快速测试脚本
用于验证项目结构是否正确
"""

def test_imports():
    """测试所有模块导入"""
    print("测试模块导入...")
    
    try:
        from palm import INet, FeatureNet, VGG
        print("✓ 模型模块导入成功")
    except ImportError as e:
        print(f"✗ 模型模块导入失败: {e}")
        return False
    
    try:
        from palm import PalmDataset, AuthDataset, ContrastivePairDataset
        print("✓ 数据集模块导入成功")
    except ImportError as e:
        print(f"✗ 数据集模块导入失败: {e}")
        return False
    
    try:
        from palm import ContrastiveLoss, TripletLoss
        print("✓ 损失函数模块导入成功")
    except ImportError as e:
        print(f"✗ 损失函数模块导入失败: {e}")
        return False
    
    try:
        from palm import load_config, get_transform, set_seed
        print("✓ 配置模块导入成功")
    except ImportError as e:
        print(f"✗ 配置模块导入失败: {e}")
        return False
    
    try:
        from palm import train_classifier, train_contrastive
        print("✓ 训练模块导入成功")
    except ImportError as e:
        print(f"✗ 训练模块导入失败: {e}")
        return False
    
    try:
        from palm import evaluate_authentication, extract_features
        print("✓ 评估模块导入成功")
    except ImportError as e:
        print(f"✗ 评估模块导入失败: {e}")
        return False
    
    return True


def test_config():
    """测试配置加载"""
    print("\n测试配置加载...")
    
    try:
        from palm import load_config, get_transform
        cfg = load_config()
        transform = get_transform(cfg)
        print("✓ 配置加载成功")
        print(f"  - 图像尺寸: {cfg['img_basic_info']['img_height']}x{cfg['img_basic_info']['img_width']}")
        print(f"  - 批次大小: {cfg['train']['batch_size']}")
        return True
    except Exception as e:
        print(f"✗ 配置加载失败: {e}")
        return False


def test_model():
    """测试模型创建"""
    print("\n测试模型创建...")
    
    try:
        import torch
        from palm import INet
        
        model = INet(feature_dim=128)
        dummy_input = torch.randn(2, 1, 128, 128)
        output = model(dummy_input)
        
        print("✓ 模型创建成功")
        print(f"  - 输入形状: {dummy_input.shape}")
        print(f"  - 输出形状: {output.shape}")
        
        assert output.shape == (2, 128), "输出形状不正确"
        print("✓ 模型输出形状正确")
        return True
    except Exception as e:
        print(f"✗ 模型测试失败: {e}")
        return False


def test_loss():
    """测试损失函数"""
    print("\n测试损失函数...")
    
    try:
        import torch
        from palm import ContrastiveLoss
        
        criterion = ContrastiveLoss(margin=0.5)
        feat1 = torch.randn(4, 128)
        feat2 = torch.randn(4, 128)
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        
        loss = criterion(feat1, feat2, labels)
        
        print("✓ 损失函数计算成功")
        print(f"  - 损失值: {loss.item():.4f}")
        return True
    except Exception as e:
        print(f"✗ 损失函数测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("="*60)
    print("掌纹识别系统 - 项目结构测试")
    print("="*60)
    
    tests = [
        ("模块导入", test_imports),
        ("配置加载", test_config),
        ("模型创建", test_model),
        ("损失函数", test_loss),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name}测试异常: {e}")
            results.append((name, False))
    
    # 总结
    print("\n" + "="*60)
    print("测试结果总结:")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{name:<20} {status}")
    
    print("="*60)
    print(f"总计: {passed}/{total} 测试通过")
    print("="*60)
    
    if passed == total:
        print("\n🎉 所有测试通过！项目结构正确。")
        print("现在可以运行: python run.py --mode all")
    else:
        print("\n⚠️  部分测试失败，请检查错误信息。")
    
    return passed == total


if __name__ == '__main__':
    import sys
    success = main()
    sys.exit(0 if success else 1)

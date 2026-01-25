class PTCChannel:
    """
    Parameter-Tuned Covert Channel (PTC^2) 模拟
    """
    @staticmethod
    def encode(parameter, message_tensor):
        # 在真实攻击中，修改参数 LSB
        # 模拟：直接返回带噪副本
        return message_tensor + torch.randn_like(message_tensor) * 0.01

    @staticmethod
    def decode(parameter):
        pass